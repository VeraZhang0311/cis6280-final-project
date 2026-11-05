"""A5: one fixed on-policy dataset, frozen image representation, one shared WM selection."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from assignment.training.training.jepa import load_model, save_model
from assignment.training.training.onpolicy_a5 import (
    ACTOR_SHA256, SCHEMA, freeze_dynamics, frozen_digest, load_fixed_dataset, ranking_metrics,
    representation_digest, rollout, sha256, write_json,
)
from assignment.training.training.train_goal_policy import code_identity


def atomic_torch_save(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def save_wm(path, model, info):
    tmp = path.with_suffix('.tmp')
    save_model(tmp, model, info)
    tmp.replace(path)


def sample_batch(data, batch, generator, device):
    states = torch.randint(len(data['ctx']), (batch,), generator=generator).numpy()
    candidates = torch.randint(data['chunks'].shape[1], (batch,), generator=generator).numpy()
    context = torch.from_numpy(data['ctx'][states]).to(device).float()
    actions = torch.from_numpy(np.concatenate([data['past'][states], data['chunks'][states, candidates]], 1)).to(device).float()
    targets = torch.from_numpy(data['tgt'][states, candidates]).to(device).float()
    return context, actions, targets


def backward_batch(model, context, actions, targets, microbatch):
    total, loss_sum = len(context), 0.
    for first in range(0, total, microbatch):
        sl = slice(first, min(total, first+microbatch))
        predictions = rollout(model, context[sl], actions[sl], grad=True)
        loss = (predictions-targets[sl].detach()).square().mean()
        weighted = loss*(len(predictions)/total)
        weighted.backward()
        loss_sum += float(weighted.detach())
    return loss_sum


@torch.no_grad()
def evaluate(model, data, device, microbatch=16):
    model.eval()
    totals = {}
    for state in range(len(data['ctx'])):
        count = data['chunks'].shape[1]
        predictions = []
        for first in range(0, count, microbatch):
            chunks = data['chunks'][state, first:first+microbatch]
            n = len(chunks)
            context = torch.from_numpy(np.repeat(data['ctx'][state:state+1], n, 0)).to(device).float()
            past = np.repeat(data['past'][state:state+1], n, 0)
            actions = torch.from_numpy(np.concatenate([past, chunks], 1)).to(device).float()
            predictions.append(rollout(model, context, actions).cpu())
        predicted = torch.cat(predictions)[None]
        target = torch.from_numpy(data['tgt'][state:state+1]).float()
        goal = torch.from_numpy(data['goal_latent'][state:state+1]).float()
        for key, value in ranking_metrics(predicted, target, goal).items():
            totals[key] = totals.get(key, 0.)+value
    return {key: value/len(data['ctx']) for key, value in totals.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True, help='Frozen 120k actor; only its identity is checked')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--microbatch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--state-every', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--wandb-project')
    p.add_argument('--wandb-entity')
    p.add_argument('--wandb-run-name', default='a5-wm-onpolicy-120k')
    p.add_argument('--wandb-run-id', default='a5-wm-onpolicy-120k-s0')
    args = p.parse_args(argv)
    attempt_started = time.monotonic()
    if min(args.steps, args.batch, args.microbatch, args.warmup, args.eval_every, args.state_every) < 1 or args.lr <= 0:
        p.error('All training counts and learning rate must be positive')
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        p.error('Use an empty checkpoint directory or --resume')
    if args.resume and not (args.out/'train-state.pt').is_file():
        p.error('--resume requires a complete recovery state')
    data, manifest, receipt = load_fixed_dataset(args.data)
    if (sha256(args.init) != manifest['plan']['models']['world_model']['sha256']
            or sha256(args.policy) != manifest['plan']['models']['actor']['sha256']
            or sha256(args.policy) != ACTOR_SHA256):
        p.error('Actor and original WM must match the fixed completed collection')
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model, original_info = load_model(args.init, args.device)
    for split, values in data.items():
        if (values['ctx'].shape[1:] != (model.cfg.context, model.cfg.dim)
                or values['chunks'].shape[-1] != model.cfg.action_dim):
            p.error(f'{split} data dimensions differ from the original WM')
    parameters = freeze_dynamics(model)
    initial_frozen, representation = frozen_digest(model), representation_digest(model)
    if any(task['representation_sha256'] != representation for task in receipt['tasks']):
        p.error('Training initialization uses a different encoder/projector than the labels')
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(.9, .999), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s+1)/args.warmup)*.5*(
        1+math.cos(math.pi*min(1., s/args.steps))))
    source = code_identity()
    meta = dict(schema=SCHEMA, code=source, actor_sha256=sha256(args.policy),
                original_wm_sha256=sha256(args.init), representation_sha256=representation,
                frozen_state_sha256=initial_frozen, dataset_manifest_sha256=sha256(args.data/'manifest.json'),
                dataset_receipt_sha256=sha256(args.data/'complete.json'),
                optimization=dict(steps=args.steps, batch=args.batch, microbatch=args.microbatch, lr=args.lr,
                                  warmup=args.warmup, betas=[.9, .999], weight_decay=.001, clip=1., seed=args.seed,
                                  eval_every=args.eval_every, state_every=args.state_every,
                                  trainable=['action_encoder', 'predictor', 'pred_proj'],
                                  objective='mean full 16-step autoregressive latent MSE; no goal loss'),
                selection=dict(rule='mean_regret_16_32', true_cost='min over future blocks 1..16',
                               std='sample std (correction=1), epsilon=1e-8', ties='earliest checkpoint',
                               includes_original_step0=True, shared_for_both_groups=True),
                counts={key: len(value['ctx']) for key, value in data.items()},
                runtime=dict(device=args.device, torch=torch.__version__, numpy=np.__version__,
                             gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None),
                wandb=dict(project=args.wandb_project, entity=args.wandb_entity,
                           name=args.wandb_run_name, id=args.wandb_run_id))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'run.json', meta) if not args.resume else None
    start, best, best_step, latest_eval = 0, None, None, None

    def checked_info(step):
        if frozen_digest(model) != initial_frozen or representation_digest(model) != representation:
            raise AssertionError('Frozen parameters or buffers changed during A5 training')
        if sha256(args.policy) != meta['actor_sha256']:
            raise AssertionError('Frozen actor file changed during A5 training')
        return dict(original_info, step=step, a5=dict(meta, selected_step=best_step,
                                                    selected_regret=best, frozen_checks=True))

    def recovery(step):
        checked_info(step)
        atomic_torch_save(args.out/'train-state.pt', dict(
            model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            step=step, best=best, best_step=best_step, latest_eval=latest_eval, meta=meta,
            data_rng=generator.get_state(), cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else []))

    if args.resume:
        state = torch.load(args.out/'train-state.pt', map_location='cpu', weights_only=False)
        if state['meta'] != meta:
            raise ValueError('Resume requires identical source, data, models, recipe and hardware')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        generator.set_state(state['data_rng'])
        torch.set_rng_state(state['cpu_rng'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start, best, best_step, latest_eval = state['step'], state['best'], state['best_step'], state['latest_eval']
        checked_info(start)
    tracker = None
    if args.wandb_project:
        import wandb
        tracker = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name,
                             id=args.wandb_run_id, resume='allow', config=meta, dir=str(args.out))
        write_json(args.out/'wandb.json', dict(url=tracker.url, id=tracker.id, project=args.wandb_project))

    def record(row):
        with (args.out/'train-log.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row), flush=True)
        if tracker is not None:
            metrics = {f'train/{k}': v for k, v in row.items() if isinstance(v, (float, int)) and k != 'step'}
            metrics.update({f'calibration/{k}': v for k, v in row.get('eval', {}).items()})
            tracker.log(metrics, step=row['step'], commit=False)

    if not args.resume:
        latest_eval = evaluate(model, data['calibration'], args.device, args.microbatch)
        best, best_step = latest_eval['selection_regret'], 0
        freeze_dynamics(model)
        record(dict(step=0, eval=latest_eval, selected=True))
        save_wm(args.out/'selected-step-000000.pt', model, checked_info(0))
        save_wm(args.out/'latest.pt', model, checked_info(0))
        recovery(0)
    running, started = [], time.monotonic()
    for step in range(start+1, args.steps+1):
        context, actions, targets = sample_batch(data['train'], args.batch, generator, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = backward_batch(model, context, actions, targets, args.microbatch)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise FloatingPointError('Nonfinite A5 loss/gradient; last atomic recovery retained')
        optimizer.step()
        scheduler.step()
        running.append(loss)
        if step % args.eval_every == 0 or step == args.steps:
            latest_eval = evaluate(model, data['calibration'], args.device, args.microbatch)
            freeze_dynamics(model)
            selected = latest_eval['selection_regret'] < best
            if selected:
                best, best_step = latest_eval['selection_regret'], step
                save_wm(args.out/f'selected-step-{step:06d}.pt', model, checked_info(step))
            save_wm(args.out/f'step-{step:06d}.pt', model, checked_info(step))
            save_wm(args.out/'latest.pt', model, checked_info(step))
            record(dict(step=step, eval=latest_eval, selected=selected, best_step=best_step))
        if step % 200 == 0 or step == args.steps:
            record(dict(step=step, loss=float(np.mean(running)), grad_norm=norm, lr=scheduler.get_last_lr()[0],
                        steps_per_s=len(running)/(time.monotonic()-started)))
            running, started = [], time.monotonic()
        if step % args.state_every == 0 or step % args.eval_every == 0 or step == args.steps:
            recovery(step)
    checked_info(args.steps)
    # Recovery state points to an immutable selected-step file; publishing is atomic.
    shutil.copyfile(args.out/f'selected-step-{best_step:06d}.pt', args.out/'model.tmp')
    (args.out/'model.tmp').replace(args.out/'model.pt')
    write_json(args.out/'selection.json', dict(step=best_step, regret=best, checkpoint='model.pt',
                                              sha256=sha256(args.out/'model.pt'), frozen_checks=True,
                                              original_retained=best_step == 0, training_complete=True,
                                              trained_steps=args.steps, last_eval=latest_eval,
                                              wall_seconds=time.monotonic()-attempt_started, wall_scope='current_attempt'))
    if tracker is not None:
        tracker.summary.update(dict(selected_step=best_step, selected_regret=best,
                                    selected_checkpoint_sha256=sha256(args.out/'model.pt'),
                                    original_retained=best_step == 0, frozen_checks=True))
        tracker.finish()


if __name__ == '__main__':
    main()
