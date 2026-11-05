"""v2 goal-conditioned actor and offline future-goal data (no deployment loop)."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from assignment.training.data.cache import RAW_ACTION_DIM, STRIDE
from assignment.training.training.policy import (
    DFPolicy, PolicyConfig, flow_matching_loss, load_policy, sample_flow,
)

MODEL_TYPE = 'goal-df-policy-v2'
GOAL_FEATURES = 'normalized_current,goal_minus_current,current_times_goal_minus_current'
HORIZON = 16
MAX_SIGMA = .04
A4_PARENT_SHA256 = 'd039fbba25eabb9b976fff945afd6b59031cbbf11df5f916a5671dd774e234bf'


def validate_a4_checkpoint(info, expected_cumulative_step=150000):
    """Require the locked tail actor; intermediate EMA evaluation needs an explicit step."""
    if (type(expected_cumulative_step) is not int
            or not 100000 < expected_cumulative_step <= 150000
            or expected_cumulative_step % 5000):
        raise ValueError('A4 checkpoint step must be an explicit saved EMA step in 105k..150k')
    if (info.get('cumulative_step') != expected_cumulative_step
            or info.get('step') != expected_cumulative_step-100000
            or not info.get('data', {}).get('include_tail', False)
            or info.get('prior_optimizer_steps') != 100000
            or info.get('init_policy', {}).get('sha256') != A4_PARENT_SHA256
            or info.get('checkpoint_selection', {}).get('rule') != 'final_step_ema'):
        raise ValueError(f'A4 requires the specified {expected_cumulative_step//1000}k '
                         'tail-coverage EMA from the locked 100k parent')


class GoalDFPolicy(DFPolicy):
    """One observation token, with a residual adapter of [goal-current, current*(goal-current)].

    Both inputs are normalized using the *same* train-split statistics before
    calling forward/loss/sample. No K, future per-token observations or WM enter
    the actor. The inherited backbone and the adapter are all trainable.
    """

    def __init__(self, cfg, adapter_hidden=None):
        if cfg.context != 1 or cfg.horizon != HORIZON:
            raise ValueError('v2 requires context=1 and horizon=16')
        super().__init__(cfg)
        self.adapter_hidden = cfg.dim if adapter_hidden is None else adapter_hidden
        if self.adapter_hidden <= 0:
            raise ValueError('adapter_hidden must be positive')
        self.goal_adapter = nn.Sequential(nn.Linear(2*cfg.latent_dim, self.adapter_hidden), nn.GELU(),
                                          nn.Linear(self.adapter_hidden, cfg.dim))
        nn.init.zeros_(self.goal_adapter[-1].weight)
        nn.init.zeros_(self.goal_adapter[-1].bias)

    @classmethod
    def from_official(cls, path, adapter_hidden=None, device='cpu'):
        original, info = load_policy(path, device)
        model = cls(original.cfg, adapter_hidden).to(device)
        # Copy only known backbone keys, leaving the new zero-initialized branch intact.
        state = model.state_dict()
        state.update(original.state_dict())
        model.load_state_dict(state, strict=True)
        return model, info

    def forward(self, current, goal, x, t):
        expected = (len(x), 1, self.cfg.latent_dim)
        if current.shape != expected or goal.shape != expected:
            raise ValueError(f'current and goal must both have shape {expected}')
        delta = goal-current
        obs = self.obs_in(current)+self.goal_adapter(torch.cat((delta, current*delta), -1))
        return self._velocity(obs, x, t)

    def loss(self, current, goal, actions, mask, generator=None):
        return flow_matching_loss(lambda x, t: self(current, goal, x, t), actions, mask, generator)

    @torch.no_grad()
    def sample(self, current, goal, steps=None, n_steps=10, lag=0, generator=None):
        return sample_flow(lambda x, t: self(current, goal, x, t), len(current), steps or self.cfg.horizon,
                           self.cfg.action_dim, current.device, n_steps, lag, generator)

    def interface(self):
        return dict(goal_features=GOAL_FEATURES, normalization='shared_train_latent_mean_std',
                    current_shape=[1, self.cfg.latent_dim], goal_shape=[1, self.cfg.latent_dim],
                    adapter=dict(input_dim=2*self.cfg.latent_dim, hidden_dim=self.adapter_hidden,
                                 output_dim=self.cfg.dim, activation='GELU', final_init='zero'),
                    sampling=dict(n_steps=10, lag=0), time_to_goal_input=False)


def save_goal_policy(path, model, info):
    torch.save(dict(model_type=MODEL_TYPE, config=asdict(model.cfg), interface=model.interface(),
                    state_dict=model.state_dict(), info=info), path)


def load_goal_policy(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get('model_type') != MODEL_TYPE:
        raise ValueError(f'Expected an explicitly typed {MODEL_TYPE} checkpoint')
    model = GoalDFPolicy(PolicyConfig(**ckpt['config']), ckpt['interface']['adapter']['hidden_dim']).to(device)
    if ckpt['interface'] != model.interface():
        raise ValueError('Unsupported goal features, normalization or adapter interface')
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return model.eval(), ckpt['info']


class FutureGoalData:
    """CPU memory maps; only selected minibatches are copied to the training device.

    Default anchors require 16 pre-stop blocks and frame t+16. include_tail=True
    admits every pre-stop anchor, samples K uniformly in 1..R, and pads/masks
    positions after R=min(16, remaining pre-stop blocks). Anchors are uniform.
    Legacy cache statistics are produced from train frames by data/encode.py.
    """

    def __init__(self, cache, latents, split, device='cpu', include_tail=False):
        if split not in ('train', 'val'):
            raise ValueError('Only the existing train and val splits may be used')
        cache, latents = Path(cache), Path(latents)
        self.index = json.loads((cache/'index.json').read_text())
        self.stats = json.loads((latents/'latents.json').read_text())
        index, stats = self.index, self.stats
        total, dim = index['frames'], stats['dim']
        if stats['frames'] != total or index['stride'] != STRIDE or index['action_dim'] != STRIDE*RAW_ACTION_DIM:
            raise ValueError('Latent/cache dimensions or stride disagree')
        self.device = torch.device(device)
        self.include_tail = include_tail
        self.z_mean = np.asarray(stats['mean'], np.float32)
        self.z_std = np.asarray(stats['std'], np.float32)
        self.a_mean = np.tile(np.asarray(index['action_mean'], np.float32), STRIDE)
        self.a_std = np.tile(np.asarray(index['action_std'], np.float32), STRIDE)
        for mean, std, size in ((self.z_mean, self.z_std, dim),
                                (self.a_mean, self.a_std, STRIDE*RAW_ACTION_DIM)):
            if (mean.shape != (size,) or std.shape != (size,) or not np.isfinite(mean).all()
                    or not np.isfinite(std).all() or (std <= 0).any()):
                raise ValueError('Invalid normalization statistics')
        anchors, remaining, selected, scene_splits, end = [], [], [], {}, 0
        for e in index['episodes']:
            # Require explicit stop metadata: the original fallback can supervise finalization.
            if not {'control_steps', 'sigma', 'scene_seed'} <= e.keys():
                raise ValueError('Explicit control_steps, sigma and scene_seed are required')
            start, length = e['start'], e['length']
            if start != end or length < 1 or e['control_steps'] < 0:
                raise ValueError('Episode frame ranges must be contiguous, ordered and valid')
            end = start+length
            scene = e['scene_seed']
            if scene in scene_splits and scene_splits[scene] != e['split']:
                raise ValueError(f'Scene {scene} leaks across data splits')
            scene_splits[scene] = e['split']
            sigma = 0. if e['sigma'] is None else e['sigma']
            if not np.isfinite(sigma) or sigma < 0:
                raise ValueError('Invalid episode noise sigma')
            n = min(e['control_steps']//STRIDE, length-1)
            full = max(0, n-HORIZON+1)
            count = n if include_tail else full
            if e['split'] == split and sigma <= MAX_SIGMA and count:
                anchors.append(np.arange(start, start+count, dtype=np.int64))
                remaining.append(np.minimum(HORIZON, n-np.arange(count)))
                selected.append(dict(episode_id=e['episode_id'], scene_seed=scene, start=start,
                                     valid_blocks=n, anchors=count, full_anchors=full, tail_anchors=count-full))
        if end != total:
            raise ValueError('Episode frame ranges disagree with cache total')
        if not anchors:
            rule = 'valid pre-stop' if include_tail else 'full-16-block'
            raise ValueError(f'No {rule} low-noise anchors in {split} split')
        self.anchors = np.concatenate(anchors)
        self.remaining = np.concatenate(remaining)
        self.r_counts = np.bincount(self.remaining, minlength=HORIZON+1)[1:]
        self.full_anchors = int(self.r_counts[-1])
        self.tail_anchors = len(self.anchors)-self.full_anchors
        self.episodes = selected
        for path, size in ((latents/'latents.f16', total*dim*2),
                           (cache/'actions.f32', total*STRIDE*RAW_ACTION_DIM*4)):
            if path.stat().st_size != size:
                raise ValueError(f'Cache file size disagrees with metadata: {path}')
        self.z = np.memmap(latents/'latents.f16', np.float16, 'r', shape=(total, dim))
        self.actions = np.memmap(cache/'actions.f32', np.float32, 'r', shape=(total, STRIDE*RAW_ACTION_DIM))

    def draw_indices(self, generator, batch):
        # CPU generator is independent of accelerator noise draws and microbatch size.
        anchors = self.anchors[torch.randint(len(self.anchors), (batch,), generator=generator).numpy()]
        offsets = torch.randint(1, HORIZON+1, (batch,), generator=generator).numpy()
        if self.include_tail:
            lengths = self.valid_lengths(anchors)
            invalid = offsets > lengths
            # Rejection from 1..16 gives an exactly uniform integer K conditional on R.
            while invalid.any():
                offsets[invalid] = torch.randint(1, HORIZON+1, (int(invalid.sum()),), generator=generator).numpy()
                invalid = offsets > lengths
        return anchors, offsets

    def valid_lengths(self, anchors):
        anchors = np.asarray(anchors)
        if anchors.ndim != 1 or not np.issubdtype(anchors.dtype, np.integer):
            raise ValueError('Expected integer anchors')
        positions = np.searchsorted(self.anchors, anchors)
        if (positions >= len(self.anchors)).any() or not np.array_equal(self.anchors[positions], anchors):
            raise ValueError('Anchor is outside the eligible split/episode range')
        return self.remaining[positions]

    def batch_at(self, anchors, offsets):
        anchors, offsets = np.asarray(anchors), np.asarray(offsets)
        if (anchors.ndim != 1 or offsets.shape != anchors.shape
                or not np.issubdtype(anchors.dtype, np.integer) or not np.issubdtype(offsets.dtype, np.integer)
                or (offsets < 1).any() or (offsets > HORIZON).any()):
            raise ValueError('Expected eligible anchors and one K in 1..16 per anchor')
        if (offsets > self.valid_lengths(anchors)).any():
            raise ValueError('Future goal K exceeds the remaining pre-stop blocks R')
        return self._batch(anchors, offsets)

    def _batch(self, anchors, offsets):
        def latent(indices):
            values = (self.z[indices].astype(np.float32)-self.z_mean)/self.z_std
            return torch.from_numpy(values[:, None]).to(self.device)
        mask = np.arange(HORIZON)[None] < self.valid_lengths(anchors)[:, None]
        if self.include_tail:
            actions = np.zeros((len(anchors), HORIZON, len(self.a_mean)), np.float32)
            row, block = np.nonzero(mask)
            # Read only valid actions, never even touch post-stop / other-episode padding slots.
            actions[row, block] = (self.actions[anchors[row]+block]-self.a_mean)/self.a_std
        else:
            blocks = anchors[:, None]+np.arange(HORIZON)
            actions = (self.actions[blocks]-self.a_mean)/self.a_std
        return (latent(anchors), latent(anchors+offsets), torch.from_numpy(actions).to(self.device),
                torch.from_numpy(mask).to(self.device))

    def sample(self, generator, batch):
        return self._batch(*self.draw_indices(generator, batch))
