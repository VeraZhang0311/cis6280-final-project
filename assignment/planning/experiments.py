"""Reproducible planning experiments. Print commands unless --run is supplied."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
COMMON = dict(generate=16, execute=1)
VARIANTS = {
    'policy': ('df', COMMON),
    'baseline': ('goal', dict(COMMON, candidates=16, tau=.25)),
    'clipbaseline': ('goal', dict(COMMON, candidates=16, tau=.25, clip_actions=True)),
    'trajectory': ('trajectory', dict(COMMON, candidates=16, adaptive=False, lookahead_weight=0.)),
    'lookahead': ('trajectory', dict(COMMON, candidates=16, adaptive=False)),
    'adaptive': ('trajectory', dict(COMMON, candidates=16, lookahead_weight=0.)),
    'combined': ('trajectory', dict(COMMON, candidates=16)),
    'budget32': ('goal', dict(COMMON, candidates=32, tau=.25)),
}


def validation_seeds():
    seeds, seed = [], 100000
    while len(seeds) < 300:
        bucket = int(hashlib.sha256(f'stacking-split-v1/{seed}'.encode()).hexdigest(), 16) % 100
        if 80 <= bucket < 90:
            seeds.append(seed)
        seed += 1
    return seeds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('smoke', 'screen', 'confirm', 'full'), default='smoke')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=['baseline', 'combined'])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--run', action='store_true')
    p.add_argument('--task-index', type=int, help='Slurm array index: one variant and one scene shard')
    p.add_argument('--shard-size', type=int, default=10)
    p.add_argument('--out', type=Path, default=ROOT/'runs/planning')
    p.add_argument('--model-root', type=Path, default=Path(os.environ.get(
        'PLANNING_MODEL_ROOT', '/vast/projects/xuhw/healthworldx/zhangrm/model/cis6280-stack-world-models-v2')))
    p.add_argument('--data-root', type=Path, default=Path(os.environ.get(
        'PLANNING_DATA_ROOT', '/vast/projects/xuhw/healthworldx/zhangrm/data/cis6280-stack-datasets-v2')))
    args = p.parse_args()
    seeds = validation_seeds()
    offset, count, sample_seeds = {
        'smoke': (0, 2, [0]), 'screen': (0, 30, [0, 2, 3]),
        'confirm': (30, 270, [0, 2, 3]), 'full': (0, 300, [0, 2, 3]),
    }[args.stage]
    suffix = ''
    if args.shard_size < 1:
        p.error('--shard-size must be positive')
    if args.task_index is not None:
        shards = math.ceil(count/args.shard_size)
        if not 0 <= args.task_index < len(args.variants)*shards:
            p.error(f'--task-index must be in [0, {len(args.variants)*shards-1}]')
        # Interleave variants so paired scene shards finish near one another.
        shard, variant_index = divmod(args.task_index, len(args.variants))
        args.variants = [args.variants[variant_index]]
        count = min(args.shard_size, count-shard*args.shard_size)
        offset += shard*args.shard_size
        suffix = f'-task{args.task_index}'
    policy = args.model_root/'policy/df-smwm-c1-sigma04/policy.pt'
    wm = args.model_root/'jepa/smwm-c3/model.pt'
    goals = args.data_root/'goals/scripted-val300-every80/goals.npz'
    commands = []
    for name in args.variants:
        kind, config = VARIANTS[name]
        command = [sys.executable, '-m', 'assignment.training.planning.evaluate', str(policy),
                   '--world-model', str(wm), '--kind', kind, '--configs', json.dumps(config),
                   '--start-seed', str(seeds[offset]), '--scenes', str(count), '--horizon', '1795',
                   '--target-count', '4', '--sample-seeds', *map(str, sample_seeds),
                   '--gpus', '0', '--workers', str(args.workers),
                   '--out-dir', str(args.out/args.stage/name)]
        if kind != 'df':
            command += ['--goals', str(goals)]
        if args.stage == 'smoke':
            command += ['--videos', '1']
        commands.append(command)
        print(shlex.join(command), flush=True)
    manifest = dict(stage=args.stage, scenes=seeds[offset:offset+count], sample_seeds=sample_seeds,
                    variants=args.variants, commands=commands, horizon=1795,
                    source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [ROOT/'assignment/training/planning/agent.py',
                                                ROOT/'assignment/training/planning/evaluate.py',
                                                Path(__file__)]},
                    status='prepared', note='No task results are implied by this manifest.')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out/f'{args.stage}{suffix}-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    if args.run:
        missing = [str(path) for path in (policy, wm, goals) if not path.is_file()]
        if missing:
            manifest.update(status='blocked', missing_artifacts=missing)
            manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
            p.error('Missing artifacts; run scripts/setup_planning.sh: '+', '.join(missing))
        manifest['status'] = 'running'
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        try:
            for command in commands:
                subprocess.run(command, cwd=ROOT, env=dict(os.environ, MUJOCO_GL='egl'), check=True)
        except (subprocess.CalledProcessError, OSError):
            manifest['status'] = 'failed'
            raise
        else:
            manifest['status'] = 'completed'
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
