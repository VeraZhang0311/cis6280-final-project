"""A5 data contracts, immutable representation checks and trajectory ranking metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from assignment.training.training.jepa import amp_dtype, autocast

SCHEMA = 'goal-actor-onpolicy-a5-v1'
ACTOR_SHA256 = '755e95f0fac13fd8990436432986584ce0f8977a23eb5c8d946546fb55477ce3'
TRAINABLE = ('action_encoder.', 'predictor.', 'pred_proj.')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def representation_digest(model):
    return state_digest({k: v for k, v in model.state_dict().items()
                         if k.startswith(('encoder.', 'projector.')) or k in ('pixel_mean', 'pixel_std')})


def freeze_dynamics(model):
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE))
    model.predictor.train()
    model.action_encoder.train()
    # pred_proj BatchNorm affine parameters may learn; its running statistics must not.
    for layer in model.pred_proj.modules():
        if isinstance(layer, torch.nn.Linear):
            layer.train()
    return [p for p in model.parameters() if p.requires_grad]


def frozen_digest(model):
    fixed = {k for k, p in model.named_parameters() if not p.requires_grad}
    fixed.update(k for k, _ in model.named_buffers())
    return state_digest({k: v for k, v in model.state_dict().items() if k in fixed})


def rollout(model, context, actions, grad=False):
    """All 16 predictions retain their graph; targets and context are never teacher forced."""
    steps = actions.shape[1]-context.shape[1]+1
    if steps != 16:
        raise ValueError('A5 requires exactly 16 future blocks')
    dtype = amp_dtype(training=grad) if context.is_cuda else torch.float32
    with torch.set_grad_enabled(grad), autocast(dtype):
        embeddings = model.action_encoder(actions)
        history, predictions = context, []
        for _ in range(steps):
            window = history[:, -model.cfg.context:]
            first = history.shape[1]-window.shape[1]
            prediction = model.predict(window, embeddings[:, first:first+window.shape[1]])[:, -1:].float()
            predictions.append(prediction)
            history = torch.cat([history, prediction], 1)
    return torch.cat(predictions, 1)


def ranking_metrics(predicted, target, goal):
    """Per-anchor full-trajectory costs. Torch std uses Stage 4's sample std convention."""
    if predicted.shape != target.shape or predicted.shape[1:3] != (32, 16):
        raise ValueError('Calibration requires the same complete 32 x 16 candidate set')
    mse = (predicted-target).square().mean((0, 1, 3))
    predicted_cost = (predicted-goal[:, None, None]).square().sum(-1).min(-1).values
    true_cost = (target-goal[:, None, None]).square().sum(-1).min(-1).values
    result = {f'h{h}': float(mse[h-1]) for h in (1, 4, 8, 16)}
    for count in (16, 32):
        cost = true_cost[:, :count]
        pick = predicted_cost[:, :count].argmin(1)
        result[f'agreement_{count}'] = float((pick == cost.argmin(1)).float().mean())
        regret = (cost.gather(1, pick[:, None])[:, 0]-cost.min(1).values)/(cost.std(1)+1e-8)
        result[f'regret_{count}'] = float(regret.mean())
    result['selection_regret'] = (result['regret_16']+result['regret_32'])/2
    return result


def validate_scene_plan(plan):
    from assignment.training.data.signals import scene_split
    if plan.get('schema') != SCHEMA:
        raise ValueError('Not an A5 scene plan')
    train, calibration = plan['scenes']['train'], plan['scenes']['calibration']
    excluded = set(plan['excluded']['development']+plan['excluded']['reserved'])
    if (not train or not calibration or len(set(train)) != len(train)
            or len(set(calibration)) != len(calibration) or set(train) & set(calibration)
            or (set(train) | set(calibration)) & excluded
            or any(scene_split(s) != 'train' for s in train+calibration)):
        raise ValueError('A5 train/calibration scenes must be unique, disjoint official train scenes')
    if any(plan['anchors'][split] < 2 or plan['anchors'][split] % 2 for split in ('train', 'calibration')):
        raise ValueError('Anchor budgets must be positive and evenly split across A4-16/A4-32')
    if not 0 < plan['keep'] <= 1 or plan['horizon'] < 90:
        raise ValueError('Invalid anchor probability or full-branch horizon')


def verify_models(policy, world_model, plan):
    if sha256(policy) != ACTOR_SHA256 or plan['models']['actor']['sha256'] != ACTOR_SHA256:
        raise ValueError('A5 requires the exact locked A4 120k actor')
    if sha256(world_model) != plan['models']['world_model']['sha256']:
        raise ValueError('A5 original WM identity differs from the A4 collection plan')


def validate_a5_wm(model, info, policy_path, wm_path, selection_path):
    meta = info.get('a5', {})
    if (sha256(policy_path) != ACTOR_SHA256 or meta.get('actor_sha256') != ACTOR_SHA256
            or meta.get('schema') != SCHEMA or not meta.get('frozen_checks')
            or meta.get('representation_sha256') != representation_digest(model)
            or meta.get('selection', {}).get('rule') != 'mean_regret_16_32'
            or meta.get('selected_step') != info.get('step')):
        raise ValueError('A5 requires the locked actor, unchanged representation and shared selected WM')
    if selection_path is None:
        raise ValueError('A5 requires the completed WM selection.json receipt')
    selection = json.loads(Path(selection_path).read_text())
    if (not selection.get('training_complete') or not selection.get('frozen_checks')
            or selection.get('sha256') != sha256(wm_path) or selection.get('step') != info.get('step')):
        raise ValueError('A5 WM does not match the completed selection receipt')
    return meta


def load_fixed_dataset(root):
    """No live shard reloads: require an atomic completion receipt and verify every byte."""
    root = Path(root)
    plan = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    receipt = json.loads((root/'complete.json').read_text(encoding='utf-8'))
    validate_scene_plan(plan['plan'])
    if receipt['manifest_sha256'] != sha256(root/'manifest.json'):
        raise ValueError('Completed dataset manifest changed')
    expected = {row['path'] for row in receipt['shards']}
    actual = {p.name for p in root.glob('*.npz')}
    if actual != expected:
        raise ValueError('Dataset shard set changed after collection')
    data = {}
    for split, count in (('train', 16), ('calibration', 32)):
        parts = []
        for row in receipt['shards']:
            if row['split'] != split:
                continue
            path = root/row['path']
            if path.parent != root or sha256(path) != row['sha256']:
                raise ValueError('Shard identity mismatch')
            with np.load(path, allow_pickle=False) as source:
                item = {key: source[key] for key in ('scene', 'sample_seed', 'step', 'goal', 'behavior',
                                                   'past', 'chunks', 'ctx', 'tgt', 'goal_latent')}
            n = len(item['scene'])
            c, d = item['ctx'].shape[1:]
            a = item['chunks'].shape[-1]
            if (item['chunks'].shape != (n, count, 16, a) or item['tgt'].shape != (n, count, 16, d)
                    or item['past'].shape != (n, c-1, a) or item['goal_latent'].shape != (n, d)
                    or not set(item['scene'].tolist()) <= set(plan['plan']['scenes'][split])
                    or not all(np.isfinite(v).all() for v in item.values())):
                raise ValueError('Malformed/incomplete A5 anchor or scene leakage')
            parts.append(item)
        if not parts:
            raise ValueError(f'Missing {split} data')
        data[split] = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        values = data[split]
        n = len(values['scene'])
        if n != plan['plan']['anchors'][split]:
            raise ValueError('Completed data does not meet the fixed anchor budget')
        if any(int((values['behavior'] == mode).sum()) != n//2 for mode in (16, 32)):
            raise ValueError('Behavior budgets must be exactly half A4-16 and half A4-32')
        ids = list(zip(values['scene'], values['sample_seed'], values['step'], values['behavior']))
        if len(set(ids)) != n:
            raise ValueError('Duplicate A5 anchor identities')
    return data, plan, receipt
