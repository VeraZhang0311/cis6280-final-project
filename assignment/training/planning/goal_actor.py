"""Frozen goal actor deployment and reproducible candidate-selection variants."""
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from assignment.training.data.cache import RAW_ACTION_DIM, STRIDE
from assignment.training.training.goal_policy import HORIZON, load_goal_policy, validate_a4_checkpoint
from assignment.training.training.jepa import amp_dtype, load_model

CANDIDATES = {'A1': 1, 'A2': 16, 'A3': 16, 'A4-16': 16, 'A4-32': 32,
              'A4-32-T1': 32, 'A4-32-T2': 32, 'A5-16': 16, 'A5-32': 32}
MODES = tuple(CANDIDATES)
SWITCH_TAU = .25
NEXT_GOAL_WEIGHT = .25
ARRIVAL_TIME_WEIGHT = .05
TERMINAL_WEIGHTS = {'A4-32-T1': .25, 'A4-32-T2': .50}
SCORE_EPS = 1e-8


@dataclass
class Selection:
    best: int
    active_min: torch.Tensor
    nearest_block: torch.Tensor
    cost: torch.Tensor
    q: torch.Tensor | None = None
    r: torch.Tensor | None = None
    eligible: torch.Tensor | None = None
    arrival: torch.Tensor | None = None
    terminal: torch.Tensor | None = None
    arrival_time_weight: float | None = None
    terminal_weight: float | None = None
    selection_rule: str = 'active_min'
    lookahead_used: bool = False
    fallback: str | None = None

    def trace(self):
        def values(tensor):
            if tensor is None:
                return None
            return tensor.detach().cpu().tolist()
        costs = self.cost.detach().cpu().tolist()
        return dict(selected_candidate=self.best, candidate_min_distance=values(self.active_min),
                    candidate_nearest_block=values(self.nearest_block),
                    candidate_arrival_block=values(self.arrival),
                    candidate_terminal_distance=values(self.terminal),
                    # Ineligible lookahead candidates have no selection cost, not JSON Infinity.
                    selection_cost=[v if np.isfinite(v) else None for v in costs],
                    q=values(self.q), r=values(self.r), eligible=values(self.eligible),
                    arrival_time_weight=self.arrival_time_weight, terminal_weight=self.terminal_weight,
                    lookahead_used=self.lookahead_used, fallback=self.fallback,
                    selection_rule='active_first_lookahead' if self.lookahead_used else self.selection_rule)


def select_candidate(active_distances, mode, threshold, next_distances=None, next_threshold=None):
    """Pure selection over *future* distances [N,16]; ties retain the earliest candidate.

    next_distances contains only each candidate's terminal distance to next goal.
    No observed state or real active index is changed here.
    """
    if mode in ('A4-16', 'A4-32', 'A5-16', 'A5-32'):
        mode = 'A2'
    if mode not in ('A2', 'A3', *TERMINAL_WEIGHTS):
        raise ValueError('Unknown candidate-selection mode')
    if (active_distances.ndim != 2 or active_distances.shape[1] != HORIZON
            or active_distances.shape[0] < 1 or not torch.isfinite(active_distances).all()
            or (active_distances < 0).any()):
        raise ValueError('Expected finite nonnegative future distances [N,16]')
    threshold = torch.as_tensor(threshold, device=active_distances.device, dtype=torch.float32)
    if threshold.numel() != 1 or not torch.isfinite(threshold).all() or (threshold < 0).any():
        raise ValueError('Invalid active goal threshold')
    minimum, nearest = active_distances.min(-1)
    selection = Selection(int(minimum.argmin()), minimum, nearest+1, minimum)
    if mode in TERMINAL_WEIGHTS:
        # Match the independently screened PARCC trajectory score. Distances are
        # normalized by the reference gap so the fixed time term has consistent
        # meaning across active goals. T2 is the predeclared stronger terminal-
        # retention variant; it differs from T1 only in terminal weight.
        gap = threshold/SWITCH_TAU
        normalized = active_distances/gap.clamp_min(SCORE_EPS)
        time_cost = ARRIVAL_TIME_WEIGHT*torch.arange(
            1, HORIZON+1, device=normalized.device, dtype=normalized.dtype)/HORIZON
        arrival_objective = normalized+time_cost
        arrival_value, arrival = arrival_objective.min(-1)
        terminal_weight = TERMINAL_WEIGHTS[mode]
        selection.cost = arrival_value+terminal_weight*normalized[:, -1]
        selection.best = int(selection.cost.argmin())
        selection.arrival = arrival+1
        selection.terminal = active_distances[:, -1]
        selection.arrival_time_weight = ARRIVAL_TIME_WEIGHT
        selection.terminal_weight = terminal_weight
        selection.selection_rule = 'trajectory_arrival_terminal'
        return selection
    if mode == 'A2':
        return selection
    if next_distances is None:
        if next_threshold is not None:
            raise ValueError('next_threshold requires next_distances')
        selection.fallback = 'last_goal'
        return selection
    if (next_distances.shape != minimum.shape or not torch.isfinite(next_distances).all()
            or (next_distances < 0).any() or next_threshold is None):
        raise ValueError('Expected finite terminal distances and next goal threshold')
    next_threshold = torch.as_tensor(next_threshold, device=active_distances.device, dtype=torch.float32)
    if next_threshold.numel() != 1 or not torch.isfinite(next_threshold).all() or (next_threshold < 0).any():
        raise ValueError('Invalid next goal threshold')
    selection.q = active_distances[:, :HORIZON-1].min(-1).values/threshold.clamp_min(SCORE_EPS)
    selection.r = next_distances/next_threshold.clamp_min(SCORE_EPS)
    selection.eligible = selection.q < 1
    if not selection.eligible.any():
        selection.fallback = 'no_eligible'
        return selection
    selection.cost = (selection.q+NEXT_GOAL_WEIGHT*selection.r).masked_fill(~selection.eligible, float('inf'))
    selection.best = int(selection.cost.argmin())
    selection.lookahead_used = True
    return selection


class GoalActorAgent:
    """step(tiled RGB) -> exactly one [5,25] raw block; always observe before replanning.

    The actor sees normalized current/active-goal latents only. World-model
    context and goal distances stay in the original frozen latent coordinates.
    last_trace is JSON-serializable and contains no full latent/action tensors.
    """

    def __init__(self, policy_path, device='cuda', mode='A1', world_model=None, seed=0,
                 a4_cumulative_step=150000, wm_selection=None, wm_evaluation_receipt=None):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.device = torch.device(device)
        self.mode = mode
        self.policy, self.info = load_goal_policy(policy_path, self.device)
        if mode.startswith('A4-'):
            validate_a4_checkpoint(self.info, a4_cumulative_step)
        if mode.startswith('A5-'):
            validate_a4_checkpoint(self.info, 120000)
        wm_path = Path(world_model or self.info['encoder'])
        if not wm_path.is_file():
            raise FileNotFoundError(f'Pass world_model with the frozen encoder checkpoint: {wm_path}')
        self.encoder, self.wm_info = load_model(wm_path, self.device)
        if mode.startswith('A5-'):
            from assignment.training.training.onpolicy_a5 import validate_a5_wm
            validate_a5_wm(self.encoder, self.wm_info, policy_path, wm_path, wm_selection, wm_evaluation_receipt)
        self.policy.eval().requires_grad_(False)
        self.encoder.eval().requires_grad_(False)
        if (self.encoder.cfg.dim != self.policy.cfg.latent_dim
                or self.encoder.cfg.action_dim != self.policy.cfg.action_dim or self.encoder.cfg.proprio_dim):
            raise ValueError('World-model and goal actor dimensions/modalities disagree')
        self.candidates = CANDIDATES[mode]
        self.generate, self.execute, self.n_steps, self.lag = HORIZON, 1, 10, 0
        self.tau, self.lookahead_weight = SWITCH_TAU, NEXT_GOAL_WEIGHT
        self._amp_dtype = amp_dtype() if self.device.type == 'cuda' else torch.float32
        self.z_mean = self._stat('z_mean', self.policy.cfg.latent_dim)
        self.z_std = self._stat('z_std', self.policy.cfg.latent_dim, positive=True)
        self.a_mean = self._stat('action_mean', STRIDE*RAW_ACTION_DIM)
        self.a_std = self._stat('action_std', STRIDE*RAW_ACTION_DIM, positive=True)
        for key, actual in (('action_mean', self.a_mean), ('action_std', self.a_std)):
            if key in self.wm_info:
                expected = torch.as_tensor(self.wm_info[key], device=self.device, dtype=torch.float32)
                if expected.shape == (RAW_ACTION_DIM,):
                    expected = expected.repeat(STRIDE)
                if not torch.equal(actual, expected):
                    raise ValueError(f'Actor and world-model {key} disagree')
        self.generator = torch.Generator(self.device).manual_seed(seed)
        self.reset()

    def _stat(self, name, size, positive=False):
        value = torch.tensor(self.info[name], dtype=torch.float32, device=self.device)
        if value.shape != (size,) or not torch.isfinite(value).all() or (positive and (value <= 0).any()):
            raise ValueError(f'Invalid actor {name}')
        return value

    def _autocast(self):
        return torch.autocast('cuda', self._amp_dtype) if self.device.type == 'cuda' else nullcontext()

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def _encode(self, frames):
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)[:, None]
        with self._autocast():
            return self.encoder.encode(x)[:, 0].float()

    @torch.inference_mode()
    def reset(self, goal_frames=None, init_frame=None, seed=None):
        encoded = None
        if goal_frames is not None:
            goals, init = np.asarray(goal_frames), np.asarray(init_frame)
            if (init_frame is None or goals.ndim != 4 or len(goals) == 0
                    or init.shape != goals.shape[1:] or goals.shape[-1] != 3
                    or goals.dtype != np.uint8 or init.dtype != np.uint8):
                raise ValueError('Expected nonempty uint8 goal RGB frames and matching official init_frame')
            encoded = self._encode(np.concatenate([init[None], goals]))
            if not torch.isfinite(encoded).all():
                raise ValueError('Nonfinite reference latents')
        elif init_frame is not None:
            raise ValueError('init_frame requires goal_frames')
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.frames, self.executed, self.index = [], [], 0
        self.goals = self.gap = self.thresholds = self.last_trace = None
        self.decision = 0
        self.setup_calls = dict(encoder=int(encoded is not None), policy=0, predictor=0)
        self.total_calls = dict(encoder=0, policy=0, policy_velocity=0, action_encoder=0, wm_rollout=0, predictor=0)
        if encoded is not None:
            self.goals = encoded[1:]
            self.gap = (encoded[1:]-encoded[:-1]).square().sum(-1)
            self.thresholds = SWITCH_TAU*self.gap

    def _advance_goal(self, current):
        switches = []
        while self.index < len(self.goals)-1:
            distance = (current-self.goals[self.index]).square().sum()
            if not distance < self.thresholds[self.index]:
                break
            switches.append(dict(previous=self.index, active=self.index+1,
                                 distance=float(distance), threshold=float(self.thresholds[self.index])))
            self.index += 1
        return switches

    @torch.inference_mode()
    def step(self, frame):
        if self.goals is None:
            raise RuntimeError('Call reset(goal_frames, init_frame, seed) before step')
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError('Expected a tiled uint8 RGB observation')
        self._sync()
        start = time.perf_counter()
        z = self._encode(frame[None])[0]
        self._sync()
        encoded_at = time.perf_counter()
        if not torch.isfinite(z).all():
            raise ValueError('Nonfinite observation latent')
        previous = self.index
        switches = self._advance_goal(z)
        active = self.goals[self.index]
        self.frames = (self.frames+[z])[-self.encoder.cfg.context:]
        n = self.candidates
        current = ((z-self.z_mean)/self.z_std)[None, None].expand(n, 1, -1)
        goal = ((active-self.z_mean)/self.z_std)[None, None].expand(n, 1, -1)
        self._sync()
        policy_start = time.perf_counter()
        with self._autocast():
            blocks = self.policy.sample(current, goal, steps=HORIZON, n_steps=10, lag=0, generator=self.generator)
        self._sync()
        policy_end = time.perf_counter()
        if blocks.shape != (n, HORIZON, STRIDE*RAW_ACTION_DIM) or not torch.isfinite(blocks).all():
            raise ValueError('Actor returned invalid action chunks')
        raw = (blocks.float()*self.a_std+self.a_mean).clamp(-1, 1)
        # WM predicts the bounded actions that would actually be executed.
        normalized = (raw-self.a_mean)/self.a_std
        calls = dict(encoder=1, policy=1, policy_velocity=10, action_encoder=0, wm_rollout=0, predictor=0)
        predictor_s = 0.
        if self.mode == 'A1':
            selected = 0
            choice = dict(selected_candidate=0, selection_rule='direct', selection_cost=None,
                          candidate_min_distance=None, candidate_nearest_block=None,
                          q=None, r=None, eligible=None, lookahead_used=False, fallback=None)
        else:
            past = self.executed[-(len(self.frames)-1):] if len(self.frames) > 1 else []
            actions = torch.cat([torch.stack(past)[None].expand(n, -1, -1), normalized], 1) if past else normalized
            self._sync()
            predictor_start = time.perf_counter()
            with self._autocast():
                imagined = self.encoder.rollout(torch.stack(self.frames)[None].expand(n, -1, -1),
                                                self.encoder.action_encoder(actions), HORIZON)
            self._sync()
            predictor_s = time.perf_counter()-predictor_start
            distances = (imagined.float()-active).square().sum(-1)
            next_distances = next_threshold = None
            if self.mode == 'A3' and self.index+1 < len(self.goals):
                next_distances = (imagined[:, -1].float()-self.goals[self.index+1]).square().sum(-1)
                next_threshold = self.thresholds[self.index+1]
            selection = select_candidate(distances, self.mode, self.thresholds[self.index],
                                         next_distances, next_threshold)
            selected, choice = selection.best, selection.trace()
            calls.update(action_encoder=1, wm_rollout=1, predictor=HORIZON)
        block = raw[selected, 0].reshape(STRIDE, RAW_ACTION_DIM).cpu().numpy().copy()
        if self.encoder.cfg.context > 1:
            self.executed = (self.executed+[normalized[selected, 0].clone()])[-(self.encoder.cfg.context-1):]
        for key, count in calls.items():
            self.total_calls[key] += count
        self.last_trace = dict(
            decision=self.decision, mode=self.mode, active_index_before=previous, active_index=self.index,
            next_index=self.index+1 if self.index+1 < len(self.goals) else None,
            goal_distance=float((z-active).square().sum()), goal_threshold=float(self.thresholds[self.index]),
            next_goal_threshold=float(self.thresholds[self.index+1]) if self.index+1 < len(self.goals) else None,
            switches=switches, candidates=n, generated_blocks=HORIZON, execute_blocks=1,
            calls=calls, **choice,
            timings_s=dict(encoder=encoded_at-start, policy=policy_end-policy_start, predictor=predictor_s))
        self._sync()
        self.last_trace['timings_s']['total'] = time.perf_counter()-start
        self.decision += 1
        return block
