"""Paired A1/A2/A3/A4 evaluation with per-decision evidence and labeled videos."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

from assignment.training.planning.evaluate import scene_goal_frames

MODES = ('A1', 'A2', 'A3', 'A4-16', 'A4-32', 'A5-16', 'A5-32')
_agents = {}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def init_worker(counter, gpus, device):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    if device == 'cuda':
        with counter.get_lock():
            rank = counter.value
            counter.value += 1
        logical = gpus[rank % len(gpus)]
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        physical = visible.split(',')[logical].strip() if visible else str(logical)
        os.environ['CUDA_VISIBLE_DEVICES'] = physical
        if physical.isdigit():
            os.environ['MUJOCO_EGL_DEVICE_ID'] = physical
    import torch
    torch.set_num_threads(1)


def make_agent(job):
    key = (job['mode'], job['policy'], job['world_model'], job['device'],
           job.get('a4_cumulative_step', 150000), job.get('wm_selection'))
    if key not in _agents:
        _agents.clear()
        from assignment.training.planning.goal_actor import GoalActorAgent
        _agents[key] = GoalActorAgent(job['policy'], job['device'], mode=job['mode'], world_model=job['world_model'],
                                     a4_cumulative_step=job.get('a4_cumulative_step', 150000),
                                     wm_selection=job.get('wm_selection'))
    return _agents[key]


def video_frame(current, goals, index, mode, scene, sample_seed, step, count, peak):
    from PIL import Image, ImageDraw
    height, width = current.shape[:2]
    panels = [('Observed', current), (f'Active goal {index}', goals[index])]
    if mode == 'A3':
        panels.append((f'Next goal {index+1}' if index+1 < len(goals) else 'Next goal: none',
                       goals[index+1] if index+1 < len(goals) else np.zeros_like(current)))
    canvas = Image.new('RGB', (width, 32+len(panels)*(height+32)), (22, 24, 29))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 1), f'{mode} scene={scene} sample={sample_seed}', fill='white')
    draw.text((5, 16), f'step={step} boxes={count} peak={peak}', fill='white')
    for panel, (label, pixels) in enumerate(panels):
        top = 32+panel*(height+32)
        draw.text((5, top+7), label, fill='white')
        canvas.paste(Image.fromarray(pixels), (0, top+32))
    return np.asarray(canvas)


def completed_episode(job):
    """The atomic episode.json is written only after logs and video finish."""
    folder = Path(job['folder'])
    marker = folder/'episode.json'
    if not job.get('resume') or not marker.is_file():
        return None
    result = json.loads(marker.read_text(encoding='utf-8'))
    expected = dict(mode=job['mode'], scene_seed=job['scene'], sample_seed=job['sample_seed'],
                    policy_seed=job['scene']+1_000_000*job['sample_seed'])
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Cached episode identity differs: {folder}')
    required = ['steps.jsonl', 'counts.jsonl']+(['video.mp4'] if job['video'] else [])
    if (not 0 < result['steps'] <= job['horizon']
            or result.get('video') != ('video.mp4' if job['video'] else None)
            or any(not (folder/name).is_file() or (folder/name).stat().st_size == 0 for name in required)):
        raise ValueError(f'Incomplete cached episode: {folder}')
    return dict(result, _reused=True)


def validate_resume(previous, current):
    for key in ('groups', 'scene_seeds', 'sample_seeds', 'split', 'policy_seed', 'models', 'actor_config',
                'actor_interface', 'training_source', 'config', 'task', 'runtime'):
        if previous[key] != current[key]:
            raise ValueError(f'Evaluation resume requires the same {key}')
    if previous['code']['source_sha256'] != current['code']['source_sha256']:
        raise ValueError('Evaluation resume requires the same source snapshot')
    if previous.get('reference') != current.get('reference'):
        raise ValueError('Evaluation resume requires the same reference run')


def stage_transition(reach, end_step, before, after, dt=.1):
    first, second = reach.get(str(before)), reach.get(str(after))
    duration = (second if second is not None else end_step)-first if first is not None else None
    return dict(at_risk=first is not None, event=second is not None,
                duration_steps=duration, duration_s=duration*dt if duration is not None else None)


def run_episode(job):
    cached = completed_episode(job)
    if cached is not None:
        return cached
    import torch
    from assignment.training.data.cache import STRIDE
    from assignment.training.data.recording import write_video
    from assignment.training.data.task import StackTaskEnv, TaskConfig
    from cube_pick.env import DT
    folder = Path(job['folder'])
    folder.mkdir(parents=True, exist_ok=job.get('resume', False))
    agent = make_agent(job)
    goals, init = scene_goal_frames(job['goals'], job['scene'])
    agent.reset(goals, init, job['scene']+1_000_000*job['sample_seed'])
    env = StackTaskEnv(TaskConfig(target_count=job['target'], horizon=job['horizon'], terminate_on_ground=False), job['scene'])
    frames, traces, counts, stage, calls = [], [], [], {}, dict(agent.total_calls)
    started = time.perf_counter()
    try:
        env.reset(('init', job['scene']))
        env.env.resolution = 128
        count = peak = env.start_count
        for k in range(1, min(count, job['target'])+1):
            stage[str(k)] = 0
        step, success, stop = 0, False, False
        with (folder/'steps.jsonl').open('w', encoding='utf-8') as stream:
            while step < job['horizon'] and not stop:
                begin = step
                frame = np.concatenate(list(env.env.render()), axis=1)
                block = agent.step(frame)
                if block.shape != (STRIDE, 25):
                    raise ValueError('Agent must return exactly one action block')
                if job['video']:
                    frames.append(video_frame(frame, goals, agent.index, job['mode'], job['scene'],
                                               job['sample_seed'], step, count, peak))
                for action in block:
                    _, _, terminated, truncated, info = env.step(action)
                    step += 1
                    count = int(info['count'])
                    peak = max(peak, count)
                    for k in range(1, min(count, job['target'])+1):
                        stage.setdefault(str(k), step)
                    counts.append(dict(control_step=step, count=count, placed=int(info['placed']),
                                       success=bool(info['success'])))
                    stop = terminated or truncated or step >= job['horizon']
                    success = bool(info['success'])
                    if stop:
                        break
                trace = dict(agent.last_trace, control_step=begin, end_control_step=step,
                             sim_time_s=begin*DT, end_sim_time_s=step*DT,
                             count_after=count, peak_count=peak, success_after=success)
                traces.append(trace)
                stream.write(json.dumps(trace, allow_nan=False)+'\n')
                stream.flush()
            if job['video']:
                frames.append(video_frame(np.concatenate(list(env.env.render()), axis=1), goals, agent.index,
                                           job['mode'], job['scene'], job['sample_seed'], step, count, peak))
        calls = dict(agent.total_calls)
        if job['mode'] == 'A1' and calls['predictor'] != 0:
            raise AssertionError('A1 must never call the WM predictor')
        with (folder/'counts.jsonl').open('w', encoding='utf-8') as stream:
            for row in counts:
                stream.write(json.dumps(row)+'\n')
        transition = stage_transition(stage, step, 1, 2, DT)
        result = dict(mode=job['mode'], scene_seed=job['scene'], sample_seed=job['sample_seed'],
                      policy_seed=job['scene']+1_000_000*job['sample_seed'], success=success,
                      steps=step, peak_boxes=peak, final_boxes=count, start_count=env.start_count,
                      first_reach_step={str(k): stage.get(str(k)) for k in range(1, job['target']+1)},
                      first_to_second=transition, third_to_fourth=stage_transition(stage, step, 3, 4, DT),
                      active_index=agent.index, goals=len(goals),
                      calls=calls, setup_calls=agent.setup_calls, seconds=time.perf_counter()-started,
                      decision_times_s=[trace['timings_s']['total'] for trace in traces],
                      lookahead_applicable=sum(trace['eligible'] is not None for trace in traces),
                      eligible_nonempty=sum(bool(trace['eligible']) and any(trace['eligible']) for trace in traces),
                      lookahead_used=sum(trace['lookahead_used'] for trace in traces),
                      decisions=len(traces), video='video.mp4' if job['video'] else None,
                      runtime=dict(device=str(agent.device), dtype=str(agent._amp_dtype),
                                   gpu=(torch.cuda.get_device_name(agent.device)
                                        if agent.device.type == 'cuda' else None),
                                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                   egl_device=os.environ.get('MUJOCO_EGL_DEVICE_ID')))
        if job['video']:
            write_video(folder/'video.mp4', [frame[None] for frame in frames], STRIDE)
        write_json(folder/'episode.json', result)
        return result
    finally:
        env.close()


def km_median(transitions):
    risk = [row for row in transitions if row['at_risk']]
    survival, remaining = 1., len(risk)
    for duration in sorted({row['duration_s'] for row in risk}):
        group = [row for row in risk if row['duration_s'] == duration]
        events = sum(row['event'] for row in group)
        survival *= 1-events/remaining
        if survival <= .5:
            return duration
        remaining -= len(group)
    return None


def summarize(episodes, target=4):
    if not episodes:
        raise ValueError('Cannot summarize an empty evaluation')
    times = [value for episode in episodes for value in episode['decision_times_s']]
    risk = [episode['first_to_second'] for episode in episodes if episode['first_to_second']['at_risk']]
    events = sum(row['event'] for row in risk)
    late = [r.get('third_to_fourth', stage_transition(r.get('first_reach_step', {}), r.get('steps', 0), 3, 4))
            for r in episodes]
    late_risk = [row for row in late if row['at_risk']]
    late_events = sum(row['event'] for row in late_risk)
    applicable = sum(row['lookahead_applicable'] for row in episodes)
    decisions = sum(row['decisions'] for row in episodes)
    return dict(episodes=len(episodes), mean_peak_boxes=float(np.mean([r['peak_boxes'] for r in episodes])),
                mean_final_boxes=float(np.mean([r['final_boxes'] for r in episodes])),
                success=float(np.mean([r['success'] for r in episodes])),
                peak_reached={str(k): float(np.mean([r['peak_boxes'] >= k for r in episodes])) for k in range(1, target+1)},
                final_reached={str(k): float(np.mean([r['final_boxes'] >= k for r in episodes])) for k in range(1, target+1)},
                mean_decision_s=float(np.mean(times)) if times else None,
                p95_decision_s=float(np.quantile(times, .95)) if times else None,
                calls={key: sum(r['calls'][key] for r in episodes) for key in episodes[0]['calls']},
                first_to_second=dict(at_risk=len(risk), events=events, censored=len(risk)-events,
                                     conversion=events/len(risk) if risk else None, km_median_s=km_median(risk)),
                third_to_fourth=dict(at_risk=len(late_risk), events=late_events, censored=len(late_risk)-late_events,
                                     conversion=late_events/len(late_risk) if late_risk else None,
                                     km_median_s=km_median(late)),
                lookahead=dict(applicable_decisions=applicable, all_decisions=decisions,
                               eligible_nonempty_rate=(sum(r['eligible_nonempty'] for r in episodes)/applicable
                                                       if applicable else None),
                               used_rate=sum(r['lookahead_used'] for r in episodes)/decisions if decisions else None))


def paired_comparisons(results, draws=2000):
    comparisons = {}
    for left, right in (('A2', 'A1'), ('A3', 'A2'), ('A4-16', 'A2'), ('A4-32', 'A4-16'),
                        ('A5-16', 'A4-16'), ('A5-32', 'A4-32'), ('A5-32', 'A5-16')):
        if left not in results or right not in results:
            continue
        a = {(r['scene_seed'], r['sample_seed']): r for r in results[left]}
        b = {(r['scene_seed'], r['sample_seed']): r for r in results[right]}
        if a.keys() != b.keys():
            raise ValueError('Paired comparison requires the same scenes and sample seeds')
        scenes = sorted({key[0] for key in a})
        metrics = {}
        for metric in ('peak_boxes', 'final_boxes', 'success'):
            per_scene = np.array([np.mean([float(a[key][metric])-float(b[key][metric])
                                          for key in a if key[0] == scene]) for scene in scenes])
            samples = np.random.default_rng(0).integers(len(scenes), size=(draws, len(scenes)))
            means = per_scene[samples].mean(-1)
            metrics[metric] = dict(difference=float(per_scene.mean()),
                                   ci95=np.quantile(means, [.025, .975]).tolist() if len(scenes) > 1 else None)
        comparisons[f'{left}_minus_{right}'] = dict(scenes=len(scenes), resampling_unit='scene', metrics=metrics)
    return comparisons


def scene_list(args):
    values = args.scenes
    if args.scene_file:
        values = json.loads(args.scene_file.read_text(encoding='utf-8-sig'))
        if isinstance(values, dict):
            values = values.get('scene_seeds', values.get('scenes'))
        if isinstance(values, list):
            values = [item['scene_seed'] if isinstance(item, dict) else item for item in values]
    if (not isinstance(values, list) or not values or any(type(value) is not int or value < 0 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError('Expected unique nonnegative integer scene seeds')
    return values


def load_a2_reference(folder, current, launch_path=None):
    """Read the completed 100k A2 reference only after checking comparable conditions."""
    from assignment.training.training.goal_policy import A4_PARENT_SHA256
    from assignment.training.training.train_goal_policy import sha256
    original = json.loads((folder/'run.json').read_text(encoding='utf-8'))
    if original['models']['goal_actor']['sha256'] != A4_PARENT_SHA256:
        raise ValueError('A4 reference A2 must use the locked 100k parent actor')
    for key in ('scene_seeds', 'sample_seeds', 'split', 'policy_seed', 'task'):
        if original[key] != current[key]:
            raise ValueError(f'A2 reference has different {key}; rerun matched-condition A2')
    for key in ('world_model', 'goals'):
        if original['models'][key]['sha256'] != current['models'][key]['sha256']:
            raise ValueError(f'A2 reference has different {key}; rerun matched-condition A2')
    for key in ('generate', 'execute', 'n_steps', 'lag', 'switch_tau', 'action_handling', 'goal_switch',
                'tie_break', 'decision_timing', 'A2'):
        if original['config'][key] != current['config'][key]:
            raise ValueError(f'A2 reference has different {key}; rerun matched-condition A2')
    if original['config']['candidates']['A2'] != 16:
        raise ValueError('Reference A2 must use 16 candidates')
    for key in ('device', 'gpus', 'workers', 'videos', 'versions'):
        if original['runtime'][key] != current['runtime'][key]:
            raise ValueError(f'A2 reference has different runtime {key}; rerun matched-condition A2')
    hardware = original['runtime'].get('gpu')
    if launch_path is not None:
        launch = json.loads(launch_path.read_text(encoding='utf-8'))
        if (launch['checkpoint_sha256'] != A4_PARENT_SHA256
                or launch['workers'] != original['runtime']['workers']):
            raise ValueError('Reference launch receipt does not identify the A2 run')
        if hardware is not None and hardware != launch['gpu']:
            raise ValueError('Reference GPU metadata is inconsistent')
        hardware = launch['gpu']
    if current['runtime']['device'] == 'cuda' and (hardware is None or hardware != current['runtime']['gpu']):
        raise ValueError('Reference GPU must be verified equal via manifest/launch receipt; otherwise rerun A2')
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((folder/'A2/episodes').glob('*/episode.json'))]
    expected = {(scene, sample) for scene in current['scene_seeds'] for sample in current['sample_seeds']}
    identities = [(row['scene_seed'], row['sample_seed']) for row in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError('Reference A2 is incomplete or has duplicate/unexpected episodes')
    if any(row['mode'] != 'A2' or row['runtime']['dtype'] != current['runtime']['dtype']
           or row['runtime']['device'] != current['runtime']['device'] for row in rows):
        raise ValueError('Reference A2 mode/device/dtype differs')
    receipt = dict(path=str(folder.resolve()), manifest_sha256=sha256(folder/'run.json'),
                   launch_sha256=sha256(launch_path) if launch_path else None, gpu=hardware,
                   episode_sha256={f"{row['scene_seed']}.{row['sample_seed']}": sha256(path)
                                   for row, path in zip(rows, sorted((folder/'A2/episodes').glob('*/episode.json')))})
    return rows, receipt


def load_a4_reference(folder, current, wm_info):
    """Reuse each same-budget A4 group only under identical inference conditions."""
    from assignment.training.training.onpolicy_a5 import ACTOR_SHA256, sha256
    original = json.loads((folder/'run.json').read_text(encoding='utf-8'))
    if (original['models']['goal_actor']['sha256'] != ACTOR_SHA256
            or current['models']['goal_actor']['sha256'] != ACTOR_SHA256
            or original['training_source']['cumulative_step'] != 120000
            or original['models']['world_model']['sha256'] != wm_info['a5']['original_wm_sha256']):
        raise ValueError('A5 reference must use the same locked 120k actor and original A4 WM')
    for key in ('scene_seeds', 'sample_seeds', 'split', 'policy_seed', 'task', 'actor_config', 'actor_interface'):
        if original[key] != current[key]:
            raise ValueError(f'A4 reference has different {key}; rerun matched-condition A4')
    if original['models']['goals']['sha256'] != current['models']['goals']['sha256']:
        raise ValueError('A4 reference has different goals')
    for key in ('generate', 'execute', 'n_steps', 'lag', 'switch_tau', 'action_handling', 'goal_switch',
                'tie_break', 'decision_timing', 'A2'):
        if original['config'][key] != current['config'][key]:
            raise ValueError(f'A4 reference has different {key}')
    for key in ('device', 'gpus', 'workers', 'videos', 'versions', 'gpu', 'dtype'):
        if original['runtime'].get(key) != current['runtime'].get(key):
            raise ValueError(f'A4 reference has different runtime {key}; rerun matched-condition A4')
    rows, hashes = {}, {}
    expected = {(s, r) for s in current['scene_seeds'] for r in current['sample_seeds']}
    for mode in current['groups']:
        baseline = mode.replace('A5-', 'A4-')
        count = int(mode.split('-')[1])
        if original['config']['candidates'][baseline] != count:
            raise ValueError('A4 reference candidate budget differs')
        paths = sorted((folder/baseline/'episodes').glob('*/episode.json'))
        values = [json.loads(p.read_text()) for p in paths]
        if (len(values) != len(expected) or {(r['scene_seed'], r['sample_seed']) for r in values} != expected
                or any(r['mode'] != baseline or r['runtime']['dtype'] != current['runtime']['dtype']
                       or r['runtime']['device'] != current['runtime']['device'] for r in values)):
            raise ValueError('A4 reference group incomplete or incompatible')
        rows[baseline] = values
        hashes[baseline] = {p.parent.name: sha256(p) for p in paths}
    return rows, dict(path=str(folder.resolve()), manifest_sha256=sha256(folder/'run.json'), episode_sha256=hashes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True, help='One locked goal actor for every group')
    parser.add_argument('--world-model', type=Path, required=True)
    parser.add_argument('--goals', type=Path, required=True)
    parser.add_argument('--official-policy', type=Path, help='Optional provenance for the original actor, not executed')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--scenes', type=int, nargs='+')
    source.add_argument('--scene-file', type=Path)
    parser.add_argument('--groups', choices=MODES, nargs='+', default=['A1', 'A2', 'A3'])
    parser.add_argument('--a4-cumulative-step', type=int, default=150000,
                        help='Explicit A4 EMA checkpoint step; default is the original final 150k')
    parser.add_argument('--reference-run', type=Path, help='Completed matched-condition 100k A2 evaluation directory')
    parser.add_argument('--reference-launch', type=Path, help='Verified launch.json for legacy references missing GPU metadata')
    parser.add_argument('--wm-selection', type=Path, help='A5 completed shared WM selection.json receipt')
    parser.add_argument('--sample-seeds', type=int, nargs='+', default=[0, 2, 3])
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Skip atomically completed episodes; retry partial episodes')
    parser.add_argument('--horizon', type=int, default=1795)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--gpus', default='0', help='Logical GPU indices within inherited CUDA_VISIBLE_DEVICES')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--videos', type=int, default=1, help='First N scenes per group, for each sample seed')
    args = parser.parse_args(argv)
    if args.reference_launch and not args.reference_run:
        parser.error('--reference-launch requires --reference-run')
    a5 = all(mode.startswith('A5-') for mode in args.groups)
    if any(mode.startswith('A5-') for mode in args.groups) and not a5:
        parser.error('Run A5 groups separately with their shared selected WM')
    if a5 and not args.wm_selection:
        parser.error('A5 requires --wm-selection')
    if args.reference_run and not (a5 or all(mode.startswith('A4-') for mode in args.groups)):
        parser.error('Reference reuse requires only A4 groups or only A5 groups')
    scenes = scene_list(args)
    if (args.horizon < 1 or args.workers < 1 or args.videos < 0 or len(set(args.groups)) != len(args.groups)
            or len(set(args.sample_seeds)) != len(args.sample_seeds) or min(args.sample_seeds) < 0):
        parser.error('Invalid horizon/workers/videos or duplicate groups/sample seeds')
    gpus = [int(value) for value in args.gpus.split(',')]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if (not gpus or min(gpus) < 0 or len(set(gpus)) != len(gpus)
            or (args.device == 'cuda' and visible and max(gpus) >= len(visible.split(',')))):
        parser.error('GPU indices must refer to the inherited visible devices')
    if args.resume and not (args.out/'run.json').is_file():
        parser.error('--resume requires the original run.json')
    if args.out.exists() and (not args.out.is_dir() or (any(args.out.iterdir()) and not args.resume)):
        parser.error('Use a new empty output directory')
    from assignment.training.data.signals import scene_split
    if any(scene_split(seed) != args.split for seed in scenes):
        parser.error('Scene list does not match the requested existing scene split')
    with np.load(args.goals) as goals:
        required = {'seeds', 'sub_frames', 'sub_count', 'init_frames'}
        if not required <= set(goals.files) or not set(scenes) <= set(goals['seeds'].tolist()):
            parser.error('Goals must contain official init frames and dense subgoals for every scene')
    from assignment.training.training.goal_policy import load_goal_policy, validate_a4_checkpoint
    from assignment.training.training.jepa import amp_dtype
    from assignment.training.training.train_goal_policy import code_identity, sha256
    import torch
    policy, training_info = load_goal_policy(args.policy)
    if any(mode.startswith('A4-') for mode in args.groups):
        validate_a4_checkpoint(training_info, args.a4_cumulative_step)
    wm_info = None
    if a5:
        from assignment.training.training.jepa import load_model
        from assignment.training.training.onpolicy_a5 import validate_a5_wm
        validate_a4_checkpoint(training_info, 120000)
        wm, wm_info = load_model(args.world_model)
        validate_a5_wm(wm, wm_info, args.policy, args.world_model, args.wm_selection)
        del wm
    actor_config, actor_interface = vars(policy.cfg), policy.interface()
    del policy
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(code=code_identity(), groups=args.groups, scene_seeds=scenes, sample_seeds=args.sample_seeds,
                    split=args.split, policy_seed='scene_seed + 1000000 * sample_seed',
                    models={key: dict(path=str(path.resolve()), sha256=sha256(path)) if path else None for key, path in
                            (('world_model', args.world_model), ('goal_actor', args.policy),
                             ('official_actor', args.official_policy), ('goals', args.goals))},
                    actor_config=actor_config, actor_interface=actor_interface,
                    training_source=dict(step=training_info.get('step'), cumulative_step=training_info.get('cumulative_step'),
                                         wandb=training_info.get('wandb'), init_policy=training_info.get('init_policy')),
                    config=dict(candidates={'A1': 1, 'A2': 16, 'A3': 16, 'A4-16': 16, 'A4-32': 32,
                                            'A5-16': 16, 'A5-32': 32},
                                generate=16, execute=1, n_steps=10,
                                lag=0, switch_tau=.25, next_goal_weight=.25, score_epsilon=1e-8,
                                action_handling='unnormalize, clip to [-1,1], renormalize for WM/history',
                                goal_switch='real observation only; strict distance < tau * reference_gap',
                                A2='min over future blocks 1..16',
                                A3='eligible q<1 over blocks 1..15; q+0.25*r; otherwise A2',
                                tie_break='earliest candidate', decision_timing='CUDA synchronized; excludes reset/render/video',
                                call_units='encoder batches; policy sample calls / velocity forwards; predictor autoregressive steps'),
                    task=dict(horizon=args.horizon, target_count=4, terminate_on_ground=False, resolution=128),
                    runtime=dict(device=args.device, gpus=gpus, workers=args.workers, videos=args.videos,
                                 gpu=torch.cuda.get_device_name(gpus[0]) if args.device == 'cuda' else None,
                                 dtype=str(amp_dtype()) if args.device == 'cuda' else 'torch.float32',
                                 versions={name: version(name) for name in ('torch', 'numpy', 'mujoco', 'imageio')}))
    if any(mode.startswith('A4-') for mode in args.groups):
        manifest['training_source']['evaluation_checkpoint'] = dict(
            cumulative_step=args.a4_cumulative_step,
            rule='final_step_ema' if args.a4_cumulative_step == 150000 else 'user_selected_intermediate_ema',
            original_training_selection=training_info.get('checkpoint_selection'))
    reference_rows = None
    reference_groups = []
    manifest['reference'] = None
    if a5:
        manifest['training_source']['wm'] = wm_info['a5']
        manifest['models']['wm_selection'] = dict(path=str(args.wm_selection.resolve()), sha256=sha256(args.wm_selection))
    if args.reference_run and a5:
        reference_rows, manifest['reference'] = load_a4_reference(args.reference_run, manifest, wm_info)
        reference_groups = list(reference_rows)
    elif args.reference_run:
        reference_rows, manifest['reference'] = load_a2_reference(args.reference_run, manifest, args.reference_launch)
        reference_rows = {'A2': reference_rows}
        reference_groups = ['A2']
    if args.resume:
        validate_resume(json.loads((args.out/'run.json').read_text(encoding='utf-8')), manifest)
    else:
        write_json(args.out/'run.json', manifest)
    with (args.out/'attempts.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(start_utc=datetime.now(timezone.utc).isoformat(), resume=args.resume,
                                     slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                                     restart_count=os.environ.get('SLURM_RESTART_COUNT', '0')))+'\n')
    ctx = mp.get_context('spawn')
    results = dict(reference_rows) if reference_rows is not None else {}
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_worker,
                             initargs=(ctx.Value('i', 0), gpus, args.device)) as pool:
        for mode in args.groups:
            jobs = [dict(mode=mode, policy=str(args.policy.resolve()), world_model=str(args.world_model.resolve()),
                         a4_cumulative_step=args.a4_cumulative_step,
                         wm_selection=str(args.wm_selection.resolve()) if args.wm_selection else None,
                         goals=str(args.goals.resolve()), device=args.device, scene=scene, sample_seed=sample,
                         target=4, horizon=args.horizon, video=i < args.videos, resume=args.resume,
                         folder=str((args.out/mode/'episodes'/f'scene-{scene}.sample-{sample}').resolve()))
                    for sample in args.sample_seeds for i, scene in enumerate(scenes)]
            results[mode] = []
            for result in pool.map(run_episode, jobs):
                results[mode].append(result)
                print(json.dumps(dict(stage='episode', mode=mode, scene=result['scene_seed'], sample=result['sample_seed'],
                                      peak=result['peak_boxes'], final=result['final_boxes'],
                                      reused=result.pop('_reused', False))), flush=True)
            write_json(args.out/mode/'summary.json', summarize(results[mode]))
    write_json(args.out/'summary.json', dict(groups={mode: summarize(rows) for mode, rows in results.items()},
                                            paired_comparisons=paired_comparisons(results),
                                            reference_groups=reference_groups,
                                            wall_s=time.perf_counter()-start, wall_scope='current_attempt'))
    print(json.dumps(dict(stage='done', output=str(args.out))), flush=True)


if __name__ == '__main__':
    main()
