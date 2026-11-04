"""Run frozen baseline diagnostics (B1 by default) on fixed scenes and seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess

import numpy as np

from assignment.training.planning.diagnostics import analyze_run, file_identity, write_json
from assignment.training.planning.evaluate import code_version, init_worker, run_episode

CONFIGS = {
    'B0': dict(generate=16, execute=1, n_steps=10, lag=0),
    'B1': dict(candidates=16, generate=16, execute=1, n_steps=10, lag=0, tau=.25),
}


def init_phase1_worker(counter, gpus):
    """Select within Slurm's allocation rather than overwriting it with physical GPU 0."""
    allowed = os.environ.get('CUDA_VISIBLE_DEVICES')
    physical = gpus
    if allowed:
        devices = allowed.split(',')
        physical = []
        for gpu in gpus:
            token = devices[gpu]
            index = token if token.isdigit() else subprocess.check_output(
                ['nvidia-smi', '-i', token, '--query-gpu=index', '--format=csv,noheader'], text=True).strip()
            physical.append(int(index))
    init_worker(counter, physical)


def source_identity():
    """Hash the actual source, including uncommitted implementation, for cluster snapshots."""
    root = Path(__file__).resolve().parents[3]
    files = sorted(list((root/'assignment').rglob('*.py'))+list((root/'cube_pick').rglob('*.py')))
    files += [root/'assignment/planning/colab.ipynb']
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return dict(git=code_version(), source_sha256=digest.hexdigest())


def validate_goals(path, scenes):
    with np.load(path) as data:
        required = {'seeds', 'sub_frames', 'init_frames', 'sub_count', 'sub_steps'}
        if not required.issubset(data.files):
            raise ValueError(f'Goals must include {sorted(required)}')
        lookup = {int(s): i for i, s in enumerate(data['seeds'])}
        for scene in scenes:
            if scene not in lookup:
                raise ValueError(f'Scene {scene} is missing from goals')
            i = lookup[scene]
            n = int(data['sub_count'][i])
            steps = data['sub_steps'][i, :n]
            # The official collector appends the final success frame off-grid.
            regular = steps[:-1] if n and steps[-1] % 80 else steps
            final_ok = n and steps[-1] > 0 and (not len(regular) or 0 <= steps[-1]-regular[-1] <= 80)
            if not final_ok or not np.array_equal(regular, np.arange(1, len(regular)+1)*80):
                raise ValueError(f'Scene {scene}: phase one requires provided 80-control-step (8s) goals')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--world-model', type=Path, required=True)
    p.add_argument('--goals', type=Path, required=True)
    p.add_argument('--scene-seeds', type=int, nargs='+', default=None)
    p.add_argument('--scene-file', type=Path, help='JSON array of fixed validation scene seeds')
    p.add_argument('--sample-seeds', type=int, nargs='+', default=[0, 2, 3])
    p.add_argument('--baselines', choices=('B0', 'B1'), nargs='+', default=['B1'],
                   help='Defaults to B1 (with planning); explicitly request B0 B1 for a paired comparison')
    p.add_argument('--out', type=Path, required=True, help='New run directory; existing runs are never overwritten')
    p.add_argument('--gpus', default='0', help='GPU indices WITHIN CUDA_VISIBLE_DEVICES, or physical ids if unset')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--videos', type=int, default=1, help='First N scenes, for selected baselines and EVERY sampler seed')
    p.add_argument('--tail-state-steps', type=int, default=56,
                   help='Keep this many final control-step contact/support snapshots; 0 disables')
    p.add_argument('--keyframes', type=int, default=6, help='Last N annotated video frames as lossless PNGs')
    args = p.parse_args()
    if len(set(args.baselines)) != len(args.baselines):
        p.error('baselines must be distinct')
    configs = {label: CONFIGS[label] for label in args.baselines}
    if args.tail_state_steps < 0 or args.keyframes < 0:
        p.error('evidence counts must be nonnegative')
    if args.scene_file and args.scene_seeds:
        p.error('choose --scene-file or --scene-seeds')
    scenes = json.loads(args.scene_file.read_text(encoding='utf-8')) if args.scene_file else (args.scene_seeds or [100038])
    if not scenes or any(type(s) is not int for s in scenes) or len(set(scenes)) != len(scenes):
        p.error('scenes must be a nonempty list of distinct integer seeds')
    from assignment.training.data.signals import scene_split
    from assignment.training.data.task import TaskConfig
    if any(scene_split(s) != 'val' for s in scenes):
        p.error('all scenes must belong to the validation split')
    if args.workers < 1 or not args.sample_seeds or len(set(args.sample_seeds)) != len(args.sample_seeds):
        p.error('workers must be positive and sampler seeds must be distinct')
    gpus = [int(g) for g in args.gpus.split(',')]
    validate_goals(args.goals, scenes)
    # Validate and hash inputs before reserving the new output directory.
    inputs = {k: file_identity(getattr(args, k)) for k in ('policy', 'world_model', 'goals')}
    args.out.mkdir(parents=True, exist_ok=False)
    task = dict(target_count=4, horizon=1795, terminate_on_ground=False)
    manifest = dict(schema_version=1, status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                    code=source_identity(), inputs=inputs, scene_seeds=scenes, sample_seeds=args.sample_seeds,
                    episode_seed_rule='scene_seed + 1000000 * sample_seed', configs=configs,
                    task=task, resolution=128, control_dt=.1, stride=5, subgoal_interval_steps=80,
                    workers=args.workers, gpus=gpus, videos=args.videos, runtime=[])
    manifest['evidence'] = dict(tail_state_steps=args.tail_state_steps, keyframes=args.keyframes,
                               keyframe_source='last block-boundary frames plus terminal observation')
    manifest['resolved_task'] = asdict(TaskConfig(**task))
    provenance = Path(__file__).resolve().parents[3]/'source-provenance.json'
    if provenance.exists():
        manifest['code']['snapshot_origin'] = json.loads(provenance.read_text(encoding='utf-8'))
    write_json(args.out/'run.json', manifest)
    jobs = []
    for label, config in configs.items():
        for sample in args.sample_seeds:
            for i, scene in enumerate(scenes):
                folder = args.out/label/'episodes'/f'scene-{scene}.sample-{sample}'
                jobs.append(('df' if label == 'B0' else 'goal', str(args.policy.resolve()),
                             str(args.world_model.resolve()), config, scene, task, 128,
                             str(folder/'video.mp4') if i < args.videos else None,
                             str(args.goals.resolve()), sample, str(folder),
                             dict(tail_state_steps=args.tail_state_steps, keyframes=args.keyframes)))
    ctx = mp.get_context('spawn')
    try:
        with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_phase1_worker,
                                 initargs=(ctx.Value('i', 0), gpus)) as pool:
            for job, result in zip(jobs, pool.map(run_episode, jobs)):
                runtime = result['runtime']
                manifest['completed_episodes'] = manifest.get('completed_episodes', 0)+1
                if runtime not in manifest['runtime']:
                    manifest['runtime'].append(runtime)
                write_json(args.out/'run.json', manifest)
                print(json.dumps(dict(kind=job[0], scene=job[4], sample=job[9], peak=result['max_placed'],
                                      final=result['final_count'], seconds=result['seconds'])), flush=True)
        summary = analyze_run(args.out)
        manifest['status'] = 'complete'
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.out/'run.json', manifest)


if __name__ == '__main__':
    main()
