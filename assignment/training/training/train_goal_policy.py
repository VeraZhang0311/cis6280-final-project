"""Finetune a flow actor on offline future goals, including masked A4 tails; keep the WM frozen.

The default recipe is fixed before training: final-step EMA checkpoint, no
selection on planning/evaluation returns. This entry point does not deploy an agent.
"""
import argparse
from contextlib import contextmanager, nullcontext
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from assignment.training.training.goal_policy import (
    FutureGoalData, GoalDFPolicy, HORIZON, MAX_SIGMA, MODEL_TYPE, load_goal_policy, save_goal_policy,
)
from assignment.training.training.jepa import amp_dtype, autocast


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity():
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True)
        return result.stdout.decode('utf-8', errors='replace').strip() if result.returncode == 0 else None
    status = git('status', '--porcelain', '--untracked-files=all')
    # Untracked implementation files are not included by git diff; hash sources too.
    sources = [*sorted((root/'assignment/training').rglob('*.py')),
               root/'scripts/train_stack_goal_policy.py']
    identity = dict(commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                dirty=status != '' if status is not None else None, status=status,
                diff_sha256=hashlib.sha256((git('diff', 'HEAD') or '').encode()).hexdigest(),
                source_sha256={p.relative_to(root).as_posix(): sha256(p) for p in sources if p.is_file()})
    provenance = root/'source-provenance.json'
    if identity['commit'] is None and provenance.is_file():
        source = json.loads(provenance.read_text())
        if source['source_sha256'] != identity['source_sha256']:
            raise ValueError('Transferred source differs from the recorded source snapshot')
        return dict(source, cluster_source_verified=True)
    return identity


def save_training_state(path, model, ema, optimizer, scheduler, data_gen, flow_gen, step, info):
    """Atomic recovery state, kept separate from the independently loadable EMA actor."""
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), data_rng=data_gen.get_state(),
                    flow_rng=flow_gen.get_state(), step=step, info=info), temporary)
    temporary.replace(path)


def precision(device):
    return autocast(amp_dtype(training=True)) if device.type == 'cuda' else nullcontext()


def empty_sampling_stats():
    return dict(samples=0, R_counts=[0]*HORIZON, K_counts=[0]*HORIZON,
                RK_counts=[[0]*HORIZON for _ in range(HORIZON)])


def record_sampling(stats, lengths, offsets):
    stats['samples'] += len(lengths)
    stats['R_counts'] = (np.asarray(stats['R_counts'])+np.bincount(lengths, minlength=HORIZON+1)[1:]).tolist()
    stats['K_counts'] = (np.asarray(stats['K_counts'])+np.bincount(offsets, minlength=HORIZON+1)[1:]).tolist()
    joint = np.bincount((lengths-1)*HORIZON+offsets-1, minlength=HORIZON*HORIZON).reshape(HORIZON, HORIZON)
    stats['RK_counts'] = (np.asarray(stats['RK_counts'])+joint).tolist()


def backward_batch(model, data, anchors, offsets, microbatch, generator):
    """Gradient of the full batch's masked block mean, including unequal tail lengths."""
    lengths = data.valid_lengths(anchors)
    valid_blocks = int(lengths.sum())
    total_loss = 0.
    for start in range(0, len(anchors), microbatch):
        end = min(start+microbatch, len(anchors))
        current, goal, actions, mask = data.batch_at(anchors[start:end], offsets[start:end])
        with precision(data.device):
            loss = model.loss(current, goal, actions, mask, generator)
        weight = int(lengths[start:end].sum())/valid_blocks
        (loss*weight).backward()
        total_loss += float(loss.detach())*weight
    return total_loss


@torch.no_grad()
def evaluate(model, data, batch, microbatch, seed=1234):
    """Fixed validation RNG, full 16-block FM and generated-action MSE; no task rollout."""
    data_gen = torch.Generator().manual_seed(seed)
    flow_gen = torch.Generator(data.device).manual_seed(seed+1)
    metrics = dict(val_loss=0., mse_h16_lag0_first=0., mse_h16_lag0_all=0.)
    anchors, offsets = data.draw_indices(data_gen, batch)
    lengths = data.valid_lengths(anchors)
    valid_blocks = int(lengths.sum())
    for start in range(0, batch, microbatch):
        end = min(start+microbatch, batch)
        current, goal, actions, mask = data.batch_at(anchors[start:end], offsets[start:end])
        with precision(data.device):
            loss = model.loss(current, goal, actions, mask, flow_gen)
            predicted = model.sample(current, goal, n_steps=10, lag=0, generator=flow_gen)
        error = (predicted-actions).square().mean(-1)
        metrics['val_loss'] += float(loss)*int(lengths[start:end].sum())/valid_blocks
        metrics['mse_h16_lag0_first'] += float(error[:, 0].sum())/batch
        metrics['mse_h16_lag0_all'] += float((error*mask).sum())/valid_blocks
    metrics['valid_blocks'] = valid_blocks
    metrics['sampled_R_counts'] = np.bincount(lengths, minlength=HORIZON+1)[1:].tolist()
    metrics['sampled_K_counts'] = np.bincount(offsets, minlength=HORIZON+1)[1:].tolist()
    return metrics


def optimizer_for(model, lr, weight_decay, device):
    decay, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 and name != 'pos' else other).append(parameter)
    return torch.optim.AdamW([dict(params=decay, weight_decay=weight_decay),
                              dict(params=other, weight_decay=0.)],
                             lr=lr, betas=(.9, .95), fused=device.type == 'cuda')


@contextmanager
def tracking(args, log_dir, info):
    """Optional online W&B tracking; local evidence and checkpoints remain authoritative."""
    if not args.wandb_project:
        yield None
        return
    import wandb
    config = {key: info[key] for key in ('model_type', 'config', 'interface', 'optimization',
                                        'rng', 'runtime', 'init_policy', 'checkpoint_selection')}
    config['data'] = {key: value for key, value in info['data'].items()
                      if key not in ('train_episodes', 'val_episodes')}
    config['code'] = {key: info['code'][key] for key in ('commit', 'branch', 'dirty')}
    config.update({key: info[key] for key in ('training_phase', 'prior_optimizer_steps', 'init_rng_state')
                   if key in info})
    tracker = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_run_name, id=args.wandb_run_id,
                         resume='allow' if args.wandb_run_id else None, mode='online',
                         job_type='actor-finetune', dir=str(log_dir.resolve()), config=config)
    info['wandb'] = dict(project=tracker.project, entity=tracker.entity, id=tracker.id,
                         url=tracker.url, sdk_version=wandb.__version__)
    tracker.define_metric('trainer/step')
    tracker.define_metric('*', step_metric='trainer/step')
    print(json.dumps(dict(stage='wandb', **info['wandb'])), flush=True)
    try:
        yield tracker
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    else:
        tracker.summary['selected_checkpoint'] = str((args.out/'policy.pt').resolve())
        tracker.summary['checkpoint_rule'] = 'final_step_ema'
        tracker.finish(exit_code=0)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--latents', type=Path, required=True)
    init = p.add_mutually_exclusive_group(required=True)
    init.add_argument('--init-policy', type=Path, help='Official unconditioned actor; never overwritten')
    init.add_argument('--init-goal-policy', type=Path, help='Goal actor EMA for a new phase; preserve its adapter')
    p.add_argument('--init-training-state', type=Path,
                   help='Continue data/flow RNG from the parent phase; optimizer and LR schedule are reset')
    p.add_argument('--encoder', type=Path, required=True,
                   help='Frozen encoder/projector source checkpoint for the latent cache (identity only)')
    p.add_argument('--out', type=Path, required=True, help='New, empty checkpoint directory')
    p.add_argument('--log-dir', type=Path, help='New, empty evidence directory; defaults to --out')
    p.add_argument('--batch', type=int, default=256, help='Effective batch per optimizer step')
    p.add_argument('--microbatch', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--ema', type=float, default=.999)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--state-every', type=int, default=1000, help='Atomic optimizer/RNG recovery interval')
    p.add_argument('--resume', action='store_true', help='Recover --out/train-state.pt after interruption')
    p.add_argument('--val-batch', type=int, default=512)
    p.add_argument('--adapter-hidden', type=int, default=None, help='Defaults to the official Transformer width')
    p.add_argument('--include-tail', action='store_true', help='A4: all valid anchors, K uniform in 1..R, masked tail padding')
    p.add_argument('--expected-init-sha256', help='Require the exact warm-start checkpoint content hash')
    p.add_argument('--expected-prior-steps', type=int, help='Require the parent cumulative optimizer step count')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda', help='cpu is supported for correctness smoke tests')
    p.add_argument('--wandb-project', help='Enable online Weights & Biases logging in this project')
    p.add_argument('--wandb-entity', help='Defaults to the authenticated W&B account')
    p.add_argument('--wandb-run-name')
    p.add_argument('--wandb-run-id', help='Stable W&B identity; does not resume optimizer state')
    return p


def validate_args(args):
    for name in ('batch', 'microbatch', 'steps', 'warmup', 'eval_every', 'save_every', 'state_every', 'val_batch'):
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if (not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay)
            or args.weight_decay < 0 or not 0 <= args.ema < 1):
        raise ValueError('Invalid optimizer or EMA settings')
    if args.microbatch > args.batch:
        raise ValueError('microbatch cannot exceed effective batch')
    if args.init_training_state and not args.init_goal_policy:
        raise ValueError('--init-training-state requires --init-goal-policy')
    if args.include_tail and (not args.init_goal_policy or args.init_training_state):
        raise ValueError('A4 tail coverage requires a goal actor warm start and fresh phase-seeded RNG')
    if args.resume and not (args.out/'train-state.pt').is_file():
        raise ValueError('--resume requires an existing train-state.pt')
    for path in (args.out, args.log_dir or args.out):
        if path.exists() and (not path.is_dir() or (any(path.iterdir()) and not args.resume)):
            raise ValueError(f'Refusing to overwrite a nonempty output directory: {path}')


def run(args):
    validate_args(args)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only cpu and cuda devices are supported')
    torch.manual_seed(args.seed)
    train = FutureGoalData(args.cache, args.latents, 'train', device, include_tail=args.include_tail)
    val = FutureGoalData(args.cache, args.latents, 'val', device, include_tail=args.include_tail)
    init_path = args.init_goal_policy or args.init_policy
    init_hash = sha256(init_path)
    if args.expected_init_sha256 and init_hash != args.expected_init_sha256:
        raise ValueError('Initialization checkpoint SHA256 differs from the locked parent')
    if args.init_goal_policy:
        model, original_info = load_goal_policy(init_path, device)
        if args.adapter_hidden is not None and args.adapter_hidden != model.adapter_hidden:
            raise ValueError('Adapter width must match the goal initialization checkpoint')
        prior_steps = original_info.get('cumulative_step', original_info['step'])
        phase = original_info.get('training_phase', 1)+1
    else:
        model, original_info = GoalDFPolicy.from_official(init_path, args.adapter_hidden, device)
        prior_steps, phase = 0, 1
    if args.expected_prior_steps is not None and prior_steps != args.expected_prior_steps:
        raise ValueError('Initialization cumulative steps differ from the locked parent')
    if model.cfg.latent_dim != train.stats['dim'] or model.cfg.action_dim != len(train.a_mean):
        raise ValueError('Official actor and cache dimensions disagree')
    # A changed normalizer would change the official actor at zero adapter output.
    for key, values in (('z_mean', train.z_mean), ('z_std', train.z_std),
                        ('action_mean', train.a_mean), ('action_std', train.a_std)):
        if key not in original_info or not np.array_equal(np.asarray(original_info[key], np.float32), values):
            raise ValueError(f'Initialization actor and cache {key} disagree')
    encoder_hash = sha256(args.encoder)
    if original_info.get('encoder_sha256', encoder_hash) != encoder_hash:
        raise ValueError('Frozen encoder source differs from the parent actor')
    cached_encoder_hash = train.stats.get('model_sha256')
    if cached_encoder_hash is not None and cached_encoder_hash != encoder_hash:
        raise ValueError('Frozen encoder identity disagrees with latent cache')
    model.train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = optimizer_for(model, args.lr, args.weight_decay, device)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., (s+1)/args.warmup)*.5*(1+math.cos(
        math.pi*min(1., s/args.steps))))
    info = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        code=code_identity(), model_type=MODEL_TYPE, interface=model.interface(), config=asdict(model.cfg),
        init_policy=dict(path=str(init_path.resolve()), sha256=init_hash,
                         model_type=MODEL_TYPE if args.init_goal_policy else 'df-policy-v1',
                         parent_step=original_info.get('step') if args.init_goal_policy else None),
        training_phase=phase, prior_optimizer_steps=prior_steps,
        init_rng_state=(dict(path=str(args.init_training_state.resolve()),
                             sha256=sha256(args.init_training_state)) if args.init_training_state else None),
        encoder=str(args.encoder.resolve()), encoder_sha256=encoder_hash,
        encoder_cache_reference=train.stats['model'],
        encoder_identity_verification='cache_sha256' if cached_encoder_hash else 'legacy_cache_metadata_only',
        # Legacy released caches carry paths, not content hashes. Record this limit explicitly.
        cache=dict(path=str(args.cache.resolve()), index_sha256=sha256(args.cache/'index.json'),
                   action_bytes=(args.cache/'actions.f32').stat().st_size),
        latents=dict(path=str(args.latents.resolve()), metadata_sha256=sha256(args.latents/'latents.json'),
                     bytes=(args.latents/'latents.f16').stat().st_size),
        z_mean=train.z_mean.tolist(), z_std=train.z_std.tolist(),
        action_mean=train.a_mean.tolist(), action_std=train.a_std.tolist(),
        data=dict(max_sigma=MAX_SIGMA, horizon=HORIZON, context=1, include_tail=args.include_tail,
                  anchor_filter=('all pre-stop blocks with a future observed state' if args.include_tail
                                 else '16 full pre-stop action blocks and observed frame t+16'),
                  anchor_distribution='uniform over all eligible anchors, no tail oversampling',
                  k_distribution=dict(kind='discrete_uniform_given_R' if args.include_tail else 'discrete_uniform',
                                      minimum=1, maximum='R' if args.include_tail else 16, actor_input=False),
                  mask='i < R; K does not truncate supervision; padding zero in normalized action space',
                  train_anchors=len(train.anchors), val_anchors=len(val.anchors),
                  train_full_anchors=train.full_anchors, train_tail_anchors=train.tail_anchors,
                  val_full_anchors=val.full_anchors, val_tail_anchors=val.tail_anchors,
                  train_R_anchor_counts=train.r_counts.tolist(), val_R_anchor_counts=val.r_counts.tolist(),
                  train_K_probabilities=[float(sum(train.r_counts[r-1]/r for r in range(k, HORIZON+1))/len(train.anchors))
                                         for k in range(1, HORIZON+1)],
                  train_episodes=train.episodes, val_episodes=val.episodes,
                  scene_split_check='no scene_seed shared across splits'),
        optimization=dict(algorithm='AdamW', betas=[.9, .95], weight_decay=args.weight_decay,
                          no_decay='pos and parameters with ndim < 2', lr=args.lr, steps=args.steps,
                          effective_batch=args.batch, microbatch=args.microbatch,
                          accumulation_steps=math.ceil(args.batch/args.microbatch),
                          accumulation_weight='microbatch_valid_blocks / effective_batch_valid_blocks',
                          schedule='min(1,(s+1)/warmup)*0.5*(1+cos(pi*min(1,s/steps)))',
                          warmup=args.warmup, gradient_clip_norm=1., ema=args.ema,
                          ema_schedule='min(ema,(1+step)/(10+step))',
                          initialization='new AdamW and LR schedule; EMA starts from the initialization actor',
                          trainable='actor backbone and goal_adapter only; no encoder or WM instantiated'),
        rng=dict(seed=args.seed, data_generator='cpu', flow_generator=str(device),
                 initialization='parent_phase_end' if args.init_training_state else 'manual_seed',
                 validation_data_seed=1234, validation_flow_seed=1235),
        checkpoint_selection=dict(rule='final_step_ema', step=args.steps,
                                  cumulative_step=prior_steps+args.steps, metric_selection=False),
        recovery=dict(state_every=args.state_every, atomic=True, includes='actor, EMA, optimizer, scheduler, RNG'),
        runtime=dict(torch=torch.__version__, numpy=np.__version__, device=str(device),
                     dtype=str(amp_dtype(training=True)) if device.type == 'cuda' else 'torch.float32',
                     gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
        parameters=sum(p.numel() for p in model.parameters()))
    info['sampling_observed'] = empty_sampling_stats()
    log_dir = args.log_dir or args.out
    for path in {args.out, log_dir}:
        path.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(stage='start', train_anchors=len(train.anchors), val_anchors=len(val.anchors),
                          effective_batch=args.batch, microbatch=args.microbatch)), flush=True)
    data_gen = torch.Generator().manual_seed(args.seed)
    flow_gen = torch.Generator(device).manual_seed(args.seed)
    if args.init_training_state and not args.resume:
        parent = torch.load(args.init_training_state, map_location='cpu', weights_only=False)
        if parent['step'] != original_info['step']:
            raise ValueError('Parent RNG state and goal checkpoint steps differ')
        if (parent['ema'].keys() != model.state_dict().keys()
                or any(not torch.equal(tensor.detach().cpu(), parent['ema'][name])
                       for name, tensor in model.state_dict().items())):
            raise ValueError('Parent RNG state does not belong to the initialization EMA actor')
        data_gen.set_state(parent['data_rng'])
        flow_gen.set_state(parent['flow_rng'])
        del parent
    first_step = 1
    if args.resume:
        state = torch.load(args.out/'train-state.pt', map_location='cpu', weights_only=False)
        for key in ('config', 'interface', 'optimization', 'rng', 'init_policy', 'init_rng_state',
                    'training_phase', 'prior_optimizer_steps', 'encoder_sha256', 'cache', 'latents',
                    'z_mean', 'z_std', 'action_mean', 'action_std', 'data'):
            if state['info'][key] != info[key]:
                raise ValueError(f'Recovery recipe differs from the original run: {key}')
        if state['info']['code']['source_sha256'] != info['code']['source_sha256']:
            raise ValueError('Cannot resume recovery state with changed training source')
        if state['info']['args'].get('wandb_run_id') != args.wandb_run_id:
            raise ValueError('Recovery must use the same W&B run identity')
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        data_gen.set_state(state['data_rng'])
        flow_gen.set_state(state['flow_rng'])
        info['sampling_observed'] = state['info']['sampling_observed']
        if 'validation_sampling' in state['info']:
            info['validation_sampling'] = state['info']['validation_sampling']
        first_step = state['step']+1
        info['resumed_from_step'] = state['step']
        del state
        print(json.dumps(dict(stage='resume', next_step=first_step)), flush=True)
    running, t0 = [], time.monotonic()
    with tracking(args, log_dir, info) as tracker, (log_dir/'train-log.jsonl').open(
            'a' if args.resume else 'w', encoding='utf-8') as log:
        for path in {args.out, log_dir}:
            (path/'run.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
        if not args.resume:
            save_training_state(args.out/'train-state.pt', model, ema, opt, sched, data_gen, flow_gen, 0, info)
        for step in range(first_step, args.steps+1):
            # Draw the whole effective batch once, keeping anchor/K sampling independent of accumulation.
            anchors, offsets = train.draw_indices(data_gen, args.batch)
            record_sampling(info['sampling_observed'], train.valid_lengths(anchors), offsets)
            opt.zero_grad(set_to_none=True)
            total_loss = backward_batch(model, train, anchors, offsets, args.microbatch, flow_gen)
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            opt.step()
            sched.step()
            with torch.no_grad():
                decay = min(args.ema, (1+step)/(10+step))
                for e, m in zip(ema.parameters(), model.parameters()):
                    e.lerp_(m, 1-decay)
            running.append((total_loss, grad))
            do_eval = step % args.eval_every == 0 or step == args.steps
            if step % 100 == 0 or do_eval:
                losses, grads = zip(*running)
                row = dict(step=step, cumulative_step=prior_steps+step,
                           loss=sum(losses)/len(losses), grad_norm=sum(grads)/len(grads),
                           lr=sched.get_last_lr()[0], steps_per_s=len(running)/(time.monotonic()-t0))
                row['sampling'] = {key: value for key, value in info['sampling_observed'].items() if key != 'RK_counts'}
                sampling_path = log_dir/'sampling.json'
                temporary = sampling_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(info['sampling_observed'])+'\n', encoding='utf-8')
                temporary.replace(sampling_path)
                if do_eval:
                    row['eval'] = evaluate(ema, val, args.val_batch, args.microbatch)
                    info['validation_sampling'] = dict(step=step, samples=args.val_batch,
                                                        valid_blocks=row['eval']['valid_blocks'],
                                                        R_counts=row['eval']['sampled_R_counts'],
                                                        K_counts=row['eval']['sampled_K_counts'])
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                print(json.dumps(row), flush=True)
                if tracker is not None:
                    metrics = {'trainer/step': step, 'trainer/total_step': prior_steps+step, 'train/loss': row['loss'],
                               'train/grad_norm': row['grad_norm'], 'train/lr': row['lr'],
                               'train/steps_per_s': row['steps_per_s']}
                    metrics.update({f'val/{key}': value for key, value in row.get('eval', {}).items()
                                    if isinstance(value, (int, float))})
                    stats = info['sampling_observed']
                    metrics['data/tail_sample_fraction'] = sum(stats['R_counts'][:-1])/stats['samples']
                    tracker.log(metrics, step=step)
                running, t0 = [], time.monotonic()
            if step % args.save_every == 0 or step == args.steps:
                save_goal_policy(args.out/f'step-{step:06d}.pt', ema,
                                 dict(info, step=step, cumulative_step=prior_steps+step))
            if step % args.state_every == 0 or step == args.steps:
                save_training_state(args.out/'train-state.pt', model, ema, opt, sched,
                                    data_gen, flow_gen, step, info)
        save_goal_policy(args.out/'policy.pt', ema,
                         dict(info, step=args.steps, cumulative_step=prior_steps+args.steps))
        for path in {args.out, log_dir}:
            (path/'run.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(stage='done', checkpoint=str(args.out/'policy.pt'), selection='final_step_ema')), flush=True)


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == '__main__':
    main()
