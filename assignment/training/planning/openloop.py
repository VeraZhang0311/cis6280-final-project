"""One terminal-selected 16-block plan, executed open-loop after a regenerated first placement.

The original cohort determines eligibility. Prefix preparation remains the unchanged B1
baseline. Only the measurement window is limited to one policy sample and one WM rollout.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from assignment.training.planning.diagnostics import annotate_frame, file_identity, write_json
from assignment.training.planning.openloop_metrics import aligned_trajectory_metrics, candidate_goal_distances
from assignment.training.planning.phase1 import CONFIGS, init_phase1_worker, source_identity

HORIZON = 16
STRIDE = 5


def as_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().float().cpu().numpy().copy()
    return np.asarray(value).copy()


class CaptureInference:
    """Observe existing calls; never invoke a model, sample, or change a returned value."""
    def __init__(self, agent):
        self.agent = agent
        self.phase = 'prefix'
        self.calls = dict(prefix_sample=0, prefix_rollout=0, test_sample=0, test_rollout=0)
        self.blocks = self.imagined = None

    def __enter__(self):
        self.original_sample = self.agent.policy.sample
        self.original_rollout = self.agent.encoder.rollout
        def sample(*args, **kwargs):
            if self.phase not in ('prefix', 'test'):
                raise RuntimeError('Policy sampling is forbidden during open-loop execution')
            if self.phase == 'test' and self.calls['test_sample']:
                raise RuntimeError('The measurement window permits only one policy sample')
            self.calls[self.phase+'_sample'] += 1
            value = self.original_sample(*args, **kwargs)
            if self.phase == 'test':
                if self.calls['test_sample'] != 1:
                    raise RuntimeError('The measurement window permits only one policy sample')
                self.blocks = as_numpy(value)
            return value
        def rollout(*args, **kwargs):
            if self.phase not in ('prefix', 'test'):
                raise RuntimeError('WM rollout is forbidden during open-loop execution')
            if self.phase == 'test' and self.calls['test_rollout']:
                raise RuntimeError('The measurement window permits only one WM rollout')
            self.calls[self.phase+'_rollout'] += 1
            value = self.original_rollout(*args, **kwargs)
            if self.phase == 'test':
                if self.calls['test_rollout'] != 1:
                    raise RuntimeError('The measurement window permits only one WM rollout')
                self.imagined = as_numpy(value)
            return value
        self.agent.policy.sample = sample
        self.agent.encoder.rollout = rollout
        return self

    def __exit__(self, *exc):
        self.agent.policy.sample = self.original_sample
        self.agent.encoder.rollout = self.original_rollout


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows), encoding='utf-8')


def save_anchor(folder, env, agent, frame, prefix_actions):
    """Preserve a real simulator checkpoint and controller history for future branching."""
    import mujoco
    from assignment.training.data.task import STATE_SPEC
    state = np.empty(mujoco.mj_stateSize(env.env.model, STATE_SPEC))
    mujoco.mj_getState(env.env.model, env.env.data, state, STATE_SPEC)
    history = list(env.env.history)
    arrays = dict(integration_state=state, actuator_targets=env.env.targets.copy(),
                  prefix_actions_raw=np.asarray(prefix_actions, np.float32),
                  sampler_state=agent.generator.get_state().cpu().numpy(),
                  goal_latents=as_numpy(agent.goals), goal_gaps=as_numpy(agent.gap),
                  agent_frames_before=np.asarray([as_numpy(z) for z in agent.frames]),
                  policy_history_before=np.asarray([as_numpy(z) for z in agent.history]),
                  executed_context_before=np.asarray([as_numpy(a) for a in agent.executed[-2:]]),
                  task_members=env.members.copy(), task_levels=env.levels.copy(), task_ground=env.ground.copy(),
                  anchor_frame=frame)
    arrays.update({'history_'+k: np.stack([row[k] for row in history]) for k in history[-1]})
    np.savez_compressed(folder/'anchor-state.npz', **arrays)
    write_json(folder/'anchor-state.json', dict(state_spec='mjSTATE_INTEGRATION', task_steps=env.steps,
               task_hold=env.hold, task_start_count=env.start_count, task_phi=env.phi,
               env_elapsed_steps=env.env.elapsed_steps, env_start_time=float(env.env.start_time),
               env_status=env.env.status, env_stop_reason=env.env.stop_reason,
               env_finalization_steps=env.env.finalization_steps, goal_index_before=agent.index,
               history_length=len(history), agent_context_frames=len(agent.frames)))


def run_case(job):
    scene, sample, prior, folder, policy, world_model, goals = job
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    base = dict(scene_seed=scene, sample_seed=sample, episode_seed=scene+1_000_000*sample,
                source_first_placement_step=prior['first_placement_step'])
    if prior['first_placement_step'] is None:
        result = dict(base, status='excluded', reason='source_episode_never_reached_first_box')
        write_json(folder/'summary.json', result)
        return result
    from assignment.training.data.signals import row_hand_parts, row_lifted
    from assignment.training.data.task import StackTaskEnv, TaskConfig
    from assignment.training.data.recording import write_video
    from assignment.training.planning.evaluate import make_agent, runtime_information, scene_goal_frames
    from assignment.training.planning.evidence import TailEvidence
    agent = make_agent('goal', policy, world_model, dict(CONFIGS['B1'], diagnostics=True))
    goal_frames, init_frame = scene_goal_frames(goals, scene)
    agent.reset(goal_frames, init_frame, base['episode_seed'])
    env = StackTaskEnv(TaskConfig(target_count=4, horizon=1795, terminate_on_ground=False), scene)
    env.reset(('init', scene))
    env.env.resolution = 128
    prefix_rows, prefix_actions, first, anchor, current, peak = [], [], None, None, 0, 0
    try:
        with CaptureInference(agent) as capture:
            for step in range(1795):
                if step % STRIDE == 0:
                    if first is not None and current >= 1:
                        if step+HORIZON*STRIDE <= 1795:
                            anchor = step
                        break
                    frame = np.concatenate(list(env.env.render()), axis=1)
                    block = agent.step(frame)
                    prefix_rows.append(dict(control_step=step, sim_time=step*.1, stacked_count=current,
                                            max_stacked_count=peak, **agent.diagnostic))
                action = block[step % STRIDE].copy()
                prefix_actions.append(action)
                _, _, terminated, truncated, info = env.step(action)
                current = int(info['count'])
                peak = max(peak, current)
                if first is None and current >= 1:
                    first = step+1
                if terminated or truncated:
                    break
            write_jsonl(folder/'prefix-steps.jsonl', prefix_rows)
            if anchor is None:
                reason = 'no_first_box_in_regenerated_prefix' if first is None else 'no_valid_full_8s_anchor'
                result = dict(base, status='excluded', reason=reason, regenerated_first_placement_step=first,
                              model_calls=capture.calls, runtime=runtime_information(agent))
                write_json(folder/'summary.json', result)
                return result
            frame0 = np.concatenate(list(env.env.render()), axis=1)
            save_anchor(folder, env, agent, frame0, prefix_actions)
            initial_count = current
            prefix_peak, peak = peak, current
            capture.phase = 'test'
            first_block = agent.step(frame0)
            capture.phase = 'observe'
            selected = int(agent.diagnostic['selected_candidate'])
            imagined, blocks = capture.imagined, capture.blocks
            if imagined.shape[:2] != (16, 16) or blocks.shape[:2] != (16, 16):
                raise ValueError('expected exactly 16 candidates, 16 future blocks')
            raw_before_clip = blocks*agent.a_std+agent.a_mean
            raw = np.clip(raw_before_clip, -1, 1).reshape(16, 16, STRIDE, 25)
            if not np.array_equal(raw[selected, 0], first_block):
                raise ValueError('captured candidate does not match the baseline first block')
            goal_index = agent.index
            goal = as_numpy(agent.goals[goal_index])
            z0 = as_numpy(agent.frames[-1])
            threshold = float(agent.tau*agent.gap[goal_index])
            initial_distance = float(np.square(z0-goal).sum())
            candidate_distances = candidate_goal_distances(imagined, goal)
            actual_frames = [frame0]
            counts, peaks = [current], [peak]
            task_rows, tail = [], TailEvidence(56)
            touched = lifted = 0
            for h in range(HORIZON):
                for j in range(STRIDE):
                    _, _, terminated, truncated, info = env.step(raw[selected, h, j])
                    relative_step = h*STRIDE+j+1
                    current = int(info['count'])
                    peak = max(peak, current)
                    row = env.env.history[-1]
                    touched += bool((row_hand_parts(row)[0] >= 2).any())
                    lifted += bool((row_lifted(row) & ~env.members).any())
                    tail.capture(env, anchor+relative_step)
                    task_rows.append(dict(control_step=anchor+relative_step, window_control_step=relative_step,
                                          stacked_count=current, max_stacked_count=peak,
                                          ground=int(info['ground']), success=bool(info['success'])))
                    if env.env.status != 'running' and relative_step != HORIZON*STRIDE:
                        raise RuntimeError('Simulator terminated before the complete fixed 16-block execution')
                actual_frames.append(np.concatenate(list(env.env.render()), axis=1))
                counts.append(current)
                peaks.append(peak)
            actual = as_numpy(agent._encode(np.asarray(actual_frames[1:])))
            observed = np.concatenate([z0[None], actual])
            metrics = aligned_trajectory_metrics(imagined[selected], actual, goal)
            clipped = ((raw_before_clip[selected] < -1) | (raw_before_clip[selected] > 1)).mean(-1)
            rows = [dict(block=h+1, window_time_s=(h+1)*.5, control_step=anchor+(h+1)*STRIDE,
                         latent_mse=float(metrics['latent_mse'][h]),
                         imagined_goal_distance_sq=float(metrics['imagined_goal_distance_sq'][h]),
                         actual_goal_distance_sq=float(metrics['actual_goal_distance_sq'][h]),
                         candidate_goal_distances_sq=candidate_distances[:, h].tolist(),
                         selected_candidate=selected, active_goal_index=goal_index,
                         goal_threshold_sq=threshold, action_clipped_fraction=float(clipped[h]),
                         stacked_count=counts[h+1], max_stacked_count=peaks[h+1]) for h in range(HORIZON)]
            np.savez_compressed(folder/'trajectories.npz', sampled_blocks_normalized=blocks,
                                raw_action_blocks=raw, imagined_latents=imagined, observed_latents=observed,
                                goal_latent=goal, candidate_goal_distances_sq=candidate_distances,
                                baseline_terminal_scores=np.asarray(agent.diagnostic['candidate_scores']),
                                **metrics)
            write_jsonl(folder/'steps.jsonl', rows)
            write_jsonl(folder/'task-steps.jsonl', task_rows)
            tail.save(folder, env)
            import imageio.v2 as imageio
            frames_dir = folder/'frames'
            frames_dir.mkdir()
            annotated = []
            for h, frame in enumerate(actual_frames):
                imageio.imwrite(frames_dir/f'block-{h:02d}.png', frame)
                annotated.append(annotate_frame(frame, goal_frames[goal_index],
                                 dict(sim_time=(anchor+h*STRIDE)*.1, stacked_count=counts[h],
                                      max_stacked_count=peaks[h], active_goal_index=goal_index),
                                 f'B1 open-loop h={h}'))
            write_video(folder/'video.mp4', [f[None] for f in annotated], STRIDE)
            if capture.calls['test_rollout'] != 1 or capture.calls['test_sample'] != 1:
                raise RuntimeError('measurement-window inference count violated')
            result = dict(base, status='complete', anchor_control_step=anchor,
                          regenerated_first_placement_step=first, initial_count=initial_count,
                          final_count=current, peak_count=peak, prefix_peak_count=prefix_peak,
                          active_goal_index=goal_index,
                          goal_count=len(goal_frames), goal_threshold_sq=threshold,
                          initial_goal_distance_sq=initial_distance, selected_candidate=selected,
                          selection='baseline terminal squared distance; per-step scores are descriptive only',
                          model_calls=capture.calls, observed_encode_batches=1,
                          mse_mean=float(metrics['latent_mse'].mean()),
                          mse_first=float(metrics['latent_mse'][0]), mse_final=float(metrics['latent_mse'][-1]),
                          imagined_goal_distance_final=float(metrics['imagined_goal_distance_sq'][-1]),
                          actual_goal_distance_final=float(metrics['actual_goal_distance_sq'][-1]),
                          action_clipped_fraction=float(clipped.mean()),
                          grasp_steps=touched, lifted_steps=lifted, runtime=runtime_information(agent),
                          seconds=round(time.perf_counter()-started, 2))
            write_json(folder/'summary.json', result)
            return result
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--gpus', default='0')
    args = p.parse_args()
    source = args.source_run.resolve()
    parent = json.loads((source/'run.json').read_text(encoding='utf-8'))
    if parent['status'] != 'complete':
        p.error('source run must be complete')
    inputs = {k: file_identity(parent['inputs'][k]['path']) for k in ('policy', 'world_model', 'goals')}
    if any(inputs[k]['sha256'] != parent['inputs'][k]['sha256'] for k in inputs):
        raise ValueError('the frozen inputs differ from the original cohort')
    args.out.mkdir(parents=True, exist_ok=False)
    summaries = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((source/'B1/episodes').glob('*/summary.json'))]
    if len(summaries) != len(parent['scene_seeds'])*len(parent['sample_seeds']):
        raise ValueError('source cohort is incomplete')
    manifest = dict(schema_version=1, status='running', source_run=file_identity(source/'run.json'),
                    code=source_identity(), inputs=inputs, total_source_cases=len(summaries),
                    source_eligible_cases=sum(s['first_placement_step'] is not None for s in summaries),
                    scene_seeds=parent['scene_seeds'], sample_seeds=parent['sample_seeds'],
                    anchor='first block boundary with current count>=1 after a regenerated first placement',
                    original_state_replay=False, prefix='unchanged B1; original no-first-box cases excluded',
                    measurement=dict(policy_samples=1, wm_rollout_calls=1, candidates=16,
                                     future_blocks=16, stride=5, seconds=8., execute_all_blocks=True,
                                     selector='last latent squared distance', fixed_active_goal=True,
                                     wm_actions='baseline sampled normalized blocks; environment uses raw clipped actions'),
                    workers=args.workers, gpus=args.gpus, runtime=[], completed_cases=0,
                    started_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.out/'run.json', manifest)
    jobs = [(s['scene_seed'], s['sample_seed'], s,
             str(args.out/'cases'/f'scene-{s["scene_seed"]}.sample-{s["sample_seed"]}'),
             inputs['policy']['path'], inputs['world_model']['path'], inputs['goals']['path']) for s in summaries]
    ctx = mp.get_context('spawn')
    results = []
    try:
        with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_phase1_worker,
                                 initargs=(ctx.Value('i', 0), [int(g) for g in args.gpus.split(',')])) as pool:
            for result in pool.map(run_case, jobs):
                results.append(result)
                manifest['completed_cases'] += 1
                if result.get('runtime') and result['runtime'] not in manifest['runtime']:
                    manifest['runtime'].append(result['runtime'])
                write_json(args.out/'run.json', manifest)
                print(json.dumps({k: result.get(k) for k in ('scene_seed', 'sample_seed', 'status', 'reason', 'mse_first', 'mse_final')}), flush=True)
        complete = [r for r in results if r['status']=='complete']
        summary = dict(source_cases=len(results), measured_cases=len(complete),
                       excluded_cases=len(results)-len(complete),
                       exclusions=dict(Counter(r['reason'] for r in results if r['status']!='complete')),
                       test_wm_rollouts=sum(r['model_calls']['test_rollout'] for r in complete),
                       selection='terminal only; no mean-score selection arm')
        write_json(args.out/'summary.json', summary)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.out/'run.json', manifest)


if __name__ == '__main__':
    main()
