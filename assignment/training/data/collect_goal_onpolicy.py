"""Collect fixed A5 real branches from A4-16/A4-32 behavior, without perturbing its RNG."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

from assignment.training.training.onpolicy_a5 import (
    SCHEMA, representation_digest, sha256, state_digest, validate_scene_plan, verify_models, write_json,
)


SIM_ATTRIBUTES = ('targets', 'elapsed_steps', 'start_time', 'status', 'stop_reason', 'finalization_steps',
                  'retreat_xy', 'lift_steps', 'withdraw_steps')


def snapshot(sim):
    import mujoco
    from assignment.training.data.recording import STATE_SPEC
    state = np.empty(mujoco.mj_stateSize(sim.model, STATE_SPEC), np.float64)
    mujoco.mj_getState(sim.model, sim.data, state, STATE_SPEC)
    return dict(state=state, attributes={name: copy.deepcopy(getattr(sim, name))
                                        for name in SIM_ATTRIBUTES if hasattr(sim, name)},
                history=copy.deepcopy(list(sim.history)), warning_number=sim.data.warning.number.copy(),
                warning_lastinfo=sim.data.warning.lastinfo.copy())


def restore(sim, saved):
    import mujoco
    from assignment.training.data.recording import STATE_SPEC
    mujoco.mj_setState(sim.model, sim.data, saved['state'], STATE_SPEC)
    mujoco.mj_forward(sim.model, sim.data)
    for name in SIM_ATTRIBUTES:
        if name in saved['attributes']:
            setattr(sim, name, copy.deepcopy(saved['attributes'][name]))
        elif hasattr(sim, name):
            delattr(sim, name)
    sim.history.clear()
    sim.history.extend(copy.deepcopy(saved['history']))
    sim.data.warning.number[:] = saved['warning_number']
    sim.data.warning.lastinfo[:] = saved['warning_lastinfo']


def execute_branches(sim, raw, encode):
    """Restore full physics/controller/history even on errors. Never finalize or pad a branch."""
    saved = snapshot(sim)
    frames = []
    try:
        for candidate in raw:
            restore(sim, saved)
            for block in candidate:
                for action in block:
                    before = sim.elapsed_steps
                    if sim.status != 'running':
                        raise ValueError('incomplete_branch_status')
                    sim.step(action)
                    if sim.status != 'running' or sim.elapsed_steps != before+1:
                        raise ValueError('incomplete_branch_time_limit_or_finalization')
                frames.append(np.concatenate(list(sim.render()), axis=1))
        return encode(np.stack(frames)).reshape(len(raw), 16, -1)
    finally:
        restore(sim, saved)


def init_worker(counter, gpus):
    from assignment.training.planning.evaluate_goal_actor import init_worker as configure
    configure(counter, gpus, 'cuda')


def collect_task(job):
    import torch
    from assignment.training.data.task import StackTaskEnv, TaskConfig
    from assignment.training.planning.evaluate import scene_goal_frames
    from assignment.training.planning.goal_actor import GoalActorAgent
    plan, out = job['plan'], Path(job['out'])
    label = f"{job['split']}-b{job['behavior']}-r{job['rank']:03d}"
    marker = out/f'{label}.json'
    if marker.is_file():
        receipt = json.loads(marker.read_text())
        if any(sha256(out/row['path']) != row['sha256'] for row in receipt['shards']):
            raise ValueError('Completed collection task shards changed')
        return receipt
    agent = GoalActorAgent(job['policy'], mode=f"A4-{job['behavior']}", world_model=job['world_model'],
                           a4_cumulative_step=120000)
    actor_before = state_digest(agent.policy.state_dict())
    representation_before = representation_digest(agent.encoder)
    completed, records = set(), []
    for path in sorted(out.glob(f'{label}-*.npz')):
        with np.load(path) as data:
            completed.update(zip(data['scene'].tolist(), data['sample_seed'].tolist(), data['step'].tolist()))
    if len(completed) > job['target']:
        raise ValueError('Existing collection exceeds locked task budget')
    shard_index = len(list(out.glob(f'{label}-*.npz')))
    begin, attempts = time.monotonic(), dict(considered=0, eligible=0, attempted=0, accepted=0, excluded={})

    def encode(frames):
        return torch.cat([agent._encode(frames[i:i+job['encode_batch']])
                          for i in range(0, len(frames), job['encode_batch'])]).cpu().numpy()

    def flush():
        nonlocal records, shard_index
        if not records:
            return
        path = out/f'{label}-{shard_index:05d}.npz'
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, **{key: np.stack([row[key] for row in records]) for key in records[0]})
        temporary.replace(path)
        shard_index += 1
        records = []

    def exclude(reason):
        attempts['excluded'][reason] = attempts['excluded'].get(reason, 0)+1

    for scene in job['scenes']:
        for sample_seed in plan['sample_seeds']:
            if len(completed) >= job['target']:
                break
            goals, initial = scene_goal_frames(job['goals'], scene)
            agent.reset(goals, initial, scene+1_000_000*sample_seed)
            env = StackTaskEnv(TaskConfig(target_count=4, horizon=plan['horizon'], terminate_on_ground=False), scene)
            try:
                env.reset(('init', scene))
                sim = env.env
                sim.resolution = 128
                for step in range(0, plan['horizon'], 5):
                    past = list(agent.executed)
                    block = agent.step(np.concatenate(list(sim.render()), axis=1))
                    attempts['considered'] += 1
                    identity = (scene, sample_seed, step)
                    # Leave a running state after all 80 control steps, not a finalization frame.
                    eligible = (len(agent.frames) == agent.encoder.cfg.context and step+80 <= plan['horizon']
                                and sim.elapsed_steps+80 < sim.max_steps and sim.status == 'running')
                    if eligible:
                        attempts['eligible'] += 1
                    rng_parts = [plan['seed'], scene, sample_seed, job['behavior'], step]
                    selected = np.random.default_rng([*rng_parts, 5106]).random() < plan['keep']
                    if eligible and selected and identity not in completed and len(completed) < job['target']:
                        attempts['attempted'] += 1
                        branch_seed = int(np.random.SeedSequence([*rng_parts, 5105]).generate_state(1)[0])
                        generator = torch.Generator(agent.device).manual_seed(branch_seed)
                        main_rng = agent.generator.get_state().clone()
                        count = plan['candidates'][job['split']]
                        current = ((agent.frames[-1]-agent.z_mean)/agent.z_std)[None, None].expand(count, 1, -1)
                        goal = ((agent.goals[agent.index]-agent.z_mean)/agent.z_std)[None, None].expand(count, 1, -1)
                        with torch.inference_mode(), agent._autocast():
                            samples = agent.policy.sample(current, goal, steps=16, n_steps=10, lag=0, generator=generator)
                            raw = (samples.float()*agent.a_std+agent.a_mean).clamp(-1, 1)
                            normalized = (raw-agent.a_mean)/agent.a_std
                        try:
                            targets = execute_branches(sim, raw.cpu().numpy().reshape(count, 16, 5, 25), encode)
                        except (RuntimeError, ValueError) as error:
                            exclude(str(error))
                        else:
                            if not np.isfinite(targets).all():
                                exclude('nonfinite_encoder_labels')
                            else:
                                records.append(dict(scene=np.int32(scene), sample_seed=np.int32(sample_seed),
                                                    step=np.int32(step), goal=np.int16(agent.index),
                                                    behavior=np.int16(job['behavior']),
                                                    branch_seed=np.uint32(branch_seed),
                                                    ctx=torch.stack(agent.frames).cpu().numpy().astype(np.float32),
                                                    past=torch.stack(past[-(agent.encoder.cfg.context-1):]).cpu().numpy().astype(np.float32),
                                                    chunks=normalized.cpu().numpy().astype(np.float32),
                                                    tgt=targets.astype(np.float32),
                                                    goal_latent=agent.goals[agent.index].cpu().numpy().astype(np.float32)))
                                completed.add(identity)
                                attempts['accepted'] += 1
                                if len(records) >= job['shard']:
                                    flush()
                        if not torch.equal(main_rng, agent.generator.get_state()):
                            raise AssertionError('Branch collection perturbed the main actor RNG')
                    if len(completed) >= job['target']:
                        break
                    stopped = False
                    for action in block[:min(5, plan['horizon']-step)]:
                        _, _, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            stopped = True
                            break
                    if stopped:
                        break
            finally:
                env.close()
        if len(completed) >= job['target']:
            break
    flush()
    if state_digest(agent.policy.state_dict()) != actor_before or representation_digest(agent.encoder) != representation_before:
        raise AssertionError('Frozen actor or image representation changed during collection')
    receipt = dict(split=job['split'], behavior=job['behavior'], rank=job['rank'],
                   states=len(completed), target=job['target'], seconds=time.monotonic()-begin,
                   actor_state_sha256=actor_before, representation_sha256=representation_before,
                   frozen_checks=True, attempt_statistics=attempts,
                   shards=[dict(path=p.name, sha256=sha256(p), split=job['split'])
                           for p in sorted(out.glob(f'{label}-*.npz'))])
    # Insufficient fixed scene order is an explicit failure; never silently change the plan.
    if len(completed) != job['target']:
        write_json(out/f'{label}-incomplete.json', receipt)
        raise RuntimeError(f'{label}: insufficient full anchors {len(completed)}/{job["target"]}; extend plan explicitly')
    write_json(marker, receipt)
    print(json.dumps(dict(stage='collection_task_done', **receipt)), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--world-model', type=Path, required=True)
    parser.add_argument('--goals', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8, help='Even number, minimum four; last two collect calibration')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--encode-batch', type=int, default=16)
    parser.add_argument('--shard', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    validate_scene_plan(plan)
    verify_models(args.policy, args.world_model, plan)
    if args.workers < 4 or args.workers % 2 or min(args.encode_batch, args.shard) < 1:
        parser.error('Use an even workers >=4 and positive encode/shard batch sizes')
    with np.load(args.goals) as source:
        if set(source['seeds'].tolist()) != set(plan['scenes']['train']+plan['scenes']['calibration']):
            parser.error('Reference goals must cover exactly the fixed train/calibration scenes')
    goal_meta = json.loads(args.goals.with_suffix('.json').read_text())
    if goal_meta['every'] != 80 or goal_meta['split'] != 'train' or goal_meta['source'] != plan['reference_source']:
        parser.error('Collection requires separate scripted train references every 80 steps')
    from assignment.training.training.train_goal_policy import code_identity
    manifest = dict(schema=SCHEMA, plan=plan, plan_sha256=sha256(args.plan), code=code_identity(),
                    goals_sha256=sha256(args.goals), goals_meta_sha256=sha256(args.goals.with_suffix('.json')),
                    workers=args.workers, encode_batch=args.encode_batch, shard=args.shard,
                    actions='clip raw [-1,1], then normalize actual executed actions; float32',
                    branches='all 16 blocks; no replanning, finalization, repeated frames or imagined labels')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'manifest.json').exists():
        previous = json.loads((args.out/'manifest.json').read_text())
        if not args.resume or previous != manifest:
            parser.error('Resume requires identical source, plan, goals and worker configuration')
    else:
        if any(args.out.iterdir()):
            parser.error('Use a new empty output directory')
        write_json(args.out/'manifest.json', manifest)
    with (args.out/'attempts.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(start_utc=datetime.now(timezone.utc).isoformat(),
                                     job=os.environ.get('SLURM_JOB_ID'), resume=args.resume))+'\n')
    jobs = []
    for split, workers in (('train', args.workers-2), ('calibration', 2)):
        per_mode = workers//2
        for behavior in (16, 32):
            for rank in range(per_mode):
                budget = plan['anchors'][split]//2
                jobs.append(dict(plan=plan, split=split, behavior=behavior, rank=rank,
                                 scenes=plan['scenes'][split][rank::per_mode],
                                 target=budget//per_mode+int(rank < budget % per_mode),
                                 policy=str(args.policy), world_model=str(args.world_model), goals=str(args.goals),
                                 out=str(args.out), encode_batch=args.encode_batch, shard=args.shard))
    ctx = mp.get_context('spawn')
    gpus = [int(value) for value in args.gpus.split(',')]
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_worker,
                             initargs=(ctx.Value('i', 0), gpus)) as pool:
        results = list(pool.map(collect_task, jobs))
    verify_models(args.policy, args.world_model, plan)
    write_json(args.out/'complete.json', dict(manifest_sha256=sha256(args.out/'manifest.json'),
                                             tasks=results, shards=[s for r in results for s in r['shards']],
                                             wall_seconds=time.monotonic()-started, wall_scope='current_attempt'))


if __name__ == '__main__':
    main()
