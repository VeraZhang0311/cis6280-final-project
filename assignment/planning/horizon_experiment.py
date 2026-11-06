"""Frozen-model 2/4/8-second planning experiment, usable from Colab or CLI.

Only WM prediction length changes. Policy generation stays at 16 blocks;
official terminal-L2 scoring, action handling and goal switching are retained.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from assignment.training.planning.agent import GoalPlanningAgent

HORIZONS = {2: 4, 4: 8, 8: 16}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def summarize(rows):
    result = []
    for seconds in sorted({r['planning_seconds'] for r in rows}):
        group = [r for r in rows if r['planning_seconds'] == seconds]
        timings = [t for r in group for t in r['decision_ms']]
        result.append(dict(planning_seconds=seconds, episodes=len(group),
                           mean_peak_boxes=float(np.mean([r['peak_boxes'] for r in group])),
                           mean_final_boxes=float(np.mean([r['final_boxes'] for r in group])),
                           success_rate=float(np.mean([r['success'] for r in group])),
                           mean_decision_ms=float(np.mean(timings)),
                           p95_decision_ms=float(np.percentile(timings, 95))))
    return result


def run_episode(agent, goal_frames, init_frame, scene, sample_seed, seconds, horizon, folder):
    # Simulator task and action loop match the official evaluation. Privileged
    # state is used only by the evaluator for metrics, never by the planner.
    from assignment.training.data.cache import STRIDE
    from assignment.training.data.recording import write_video
    from assignment.training.data.task import StackTaskEnv, TaskConfig
    agent.reset(goal_frames, init_frame, seed=int(scene) + 1_000_000 * int(sample_seed))
    env = StackTaskEnv(TaskConfig(target_count=4, horizon=horizon, terminate_on_ground=False), int(scene))
    frames, times, peak, info, success = [], [], 0, {}, False
    start = time.perf_counter()
    try:
        env.reset(('init', int(scene)))
        env.env.resolution = 128
        with (folder / 'decisions.jsonl').open('w') as trace:
            for step in range(horizon):
                if step % STRIDE == 0:
                    frame = np.concatenate(list(env.env.render()), axis=1)
                    if agent._device_type == 'cuda':
                        torch.cuda.synchronize(agent.device)
                    t0 = time.perf_counter()
                    block = agent.step(frame)
                    if agent._device_type == 'cuda':
                        torch.cuda.synchronize(agent.device)
                    elapsed = 1000 * (time.perf_counter() - t0)
                    times.append(elapsed)
                    if block.shape != (5, 25) or block.dtype != np.float32 or not np.isfinite(block).all() or np.abs(block).max() > 1:
                        raise ValueError('Invalid planner action block')
                    frames.append(np.concatenate([frame, goal_frames[agent.index]], axis=0))
                    trace.write(json.dumps(dict(control_step=step, goal_index=agent.index,
                                                decision_ms=elapsed,
                                                candidate_costs=agent.last_distance.cpu().tolist())) + '\n')
                    trace.flush()
                    if step % 200 == 0:
                        print(f'{seconds}s / scene {scene} / seed {sample_seed}: step {step}/{horizon}, peak {peak}', flush=True)
                _, _, terminated, truncated, info = env.step(block[step % STRIDE])
                peak = max(peak, int(info['placed']))
                if terminated or truncated:
                    success = bool(info.get('success', False))
                    break
    finally:
        env.close()
    video = folder / 'video.mp4'
    write_video(video, [f[None] for f in frames], STRIDE)
    row = dict(planning_seconds=seconds, planning_blocks=HORIZONS[seconds], scene=int(scene),
               sample_seed=int(sample_seed), control_steps=step + 1,
               peak_boxes=peak, final_boxes=int(info.get('count', 0)), success=success,
               decision_ms=times, wall_seconds=time.perf_counter() - start, video=str(video))
    write_json(folder / 'episode.json', row)  # Completion marker written last.
    print(f'Finished {seconds}s: peak {peak}, final {row["final_boxes"]}, success {success}', flush=True)
    return row


def run_experiment(policy, world_model, goals, out, scenes=(100038,), sample_seeds=(0,),
                   planning_seconds=(2, 4, 8), episode_steps=1795, device='cuda', resume=True):
    """Run a small paired comparison. Short episodes are implementation checks only."""
    from assignment.training.data.signals import scene_split
    scenes, sample_seeds, planning_seconds = list(scenes), list(sample_seeds), list(planning_seconds)
    if not scenes or not sample_seeds or not planning_seconds:
        raise ValueError('Scenes, sample seeds and horizons must be nonempty')
    if any(len(v) != len(set(v)) for v in (scenes, sample_seeds, planning_seconds)):
        raise ValueError('Duplicate scenes, seeds or horizons')
    if any(s not in HORIZONS for s in planning_seconds) or not 1 <= episode_steps <= 1795:
        raise ValueError('Use 2/4/8 seconds and an episode length from 1 to 1795')
    if any(scene_split(int(s)) != 'val' for s in scenes):
        raise ValueError('All scenes must be in the official validation split')
    with np.load(goals) as source:
        data = {k: source[k] for k in source.files}
    index = {int(s): i for i, s in enumerate(data['seeds'])}
    if any(int(s) not in index for s in scenes):
        raise ValueError('A requested scene is missing from official goals')
    inputs = {name: dict(path=str(Path(path).resolve()), sha256=digest(path))
              for name, path in dict(policy=policy, world_model=world_model, goals=goals).items()}
    # Hash the actual source even when Colab installs it into the downloaded code tree.
    source_paths = [Path(__file__), Path(__file__).parents[1] / 'training/planning/agent.py']
    manifest = dict(protocol='shafaa-horizon-v1', inputs=inputs,
                    sources={p.name: digest(p) for p in source_paths},
                    scenes=scenes, sample_seeds=sample_seeds, planning_seconds=planning_seconds,
                    episode_steps=episode_steps, smoke_only=episode_steps != 1795,
                    generate=16, candidates=16, execute=1, n_steps=10, lag=0, tau=.25,
                    scoring='terminal_squared_L2', prediction_action_handling='official_baseline',
                    device=str(device), python=platform.python_version(), torch=torch.__version__,
                    numpy=np.__version__, gpu=(torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / 'run.json'
    if receipt.exists():
        if not resume or json.loads(receipt.read_text()) != manifest:
            raise ValueError('Output exists with different settings/source. Use a fresh output folder.')
    else:
        if any(out.iterdir()):
            raise ValueError('Output folder is nonempty without a run manifest')
        write_json(receipt, manifest)
    agent = GoalPlanningAgent(str(policy), device=device, world_model=str(world_model),
                              candidates=16, generate=16, execute=1, n_steps=10, lag=0, tau=.25)
    agent.policy.eval().requires_grad_(False)
    agent.encoder.eval().requires_grad_(False)
    rows = []
    for seconds in planning_seconds:
        agent.planning_horizon = HORIZONS[seconds]
        for scene in scenes:
            i = index[int(scene)]
            goal_frames = data['sub_frames'][i, :int(data['sub_count'][i])]
            for seed in sample_seeds:
                folder = out / f'horizon-{seconds}s' / f'scene-{scene}.sample-{seed}'
                folder.mkdir(parents=True, exist_ok=True)
                marker = folder / 'episode.json'
                if marker.exists() and (folder / 'video.mp4').is_file():
                    row = json.loads(marker.read_text())
                    print(f'Reusing completed {seconds}s / {scene} / {seed}', flush=True)
                else:
                    row = run_episode(agent, goal_frames, data['init_frames'][i], scene, seed,
                                      seconds, episode_steps, folder)
                rows.append(row)
                write_json(out / 'summary.json', dict(smoke_only=episode_steps != 1795, groups=summarize(rows)))
    for name, path in dict(policy=policy, world_model=world_model, goals=goals).items():
        if digest(path) != inputs[name]['sha256']:
            raise RuntimeError(f'Input file changed during experiment: {name}')
    print(json.dumps(summarize(rows), indent=2), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', required=True)
    p.add_argument('--world-model', required=True)
    p.add_argument('--goals', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--scenes', type=int, nargs='+', default=[100038])
    p.add_argument('--sample-seeds', type=int, nargs='+', default=[0])
    p.add_argument('--planning-seconds', type=int, nargs='+', default=[2, 4, 8])
    p.add_argument('--episode-steps', type=int, default=1795)
    p.add_argument('--device', default='cuda')
    args = vars(p.parse_args())
    run_experiment(**args)


if __name__ == '__main__':
    main()
