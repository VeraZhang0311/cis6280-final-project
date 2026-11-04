"""Closed-loop policy agent and goal-conditioned planning agent."""
import contextlib
from pathlib import Path

import numpy as np
import torch

from assignment.training.data.cache import RAW_ACTION_DIM, STRIDE
from assignment.training.training.jepa import amp_dtype
from assignment.training.training.policy import load_policy


class LatentPolicyAgent:
    """Closed-loop controller that samples action chunks from the diffusion-forcing policy.

    Call step(frame) at every block boundary (every STRIDE=5 control steps) with the tiled
    (128, 256, 3) uint8 RGB frame; returns the (STRIDE, 25) float32 raw actions in [-1, 1]
    for that block.
    """

    def __init__(self, policy_path, device='cuda', generate=None, execute=1, n_steps=10, lag=0, seed=0,
                 world_model=None, diagnostics=False):
        from assignment.training.training.jepa import load_model
        self.policy, info = load_policy(policy_path, device)
        wm_path = Path(world_model or info['encoder'])
        if not wm_path.exists():
            raise FileNotFoundError(
                f"World-model checkpoint not found at '{wm_path}'. Pass world_model='path/to/model.pt' "
                f"or set ckpt['info']['encoder'] in '{policy_path}' to your local world-model file."
            )
        self.encoder, _ = load_model(wm_path, device)
        self.world_model_path = str(wm_path.resolve())
        self.diagnostics = diagnostics
        self.device, self.info = str(device), info
        self._device_type = torch.device(device).type
        self._amp_dtype = amp_dtype()
        self.generate = generate or self.policy.cfg.horizon
        self.execute, self.n_steps, self.lag = execute, n_steps, lag
        if diagnostics and execute != 1:
            raise ValueError('phase-1 diagnostics require execute=1 for one-block alignment')
        if not 1 <= execute <= self.generate <= self.policy.cfg.horizon:
            raise ValueError('need 1 <= execute <= generate <= horizon')
        self.z_mean = torch.tensor(info['z_mean'], device=device)
        self.z_std = torch.tensor(info['z_std'], device=device)
        self.a_mean = np.asarray(info['action_mean'], np.float32)
        self.a_std = np.asarray(info['action_std'], np.float32)
        self.generator = torch.Generator(device).manual_seed(seed)
        self.reset()

    def _autocast(self):
        if self._device_type == 'cuda':
            return torch.autocast('cuda', self._amp_dtype)
        return contextlib.nullcontext()

    def reset(self, seed=None):
        """Start an episode. Passing a seed (e.g. the scene seed) makes the episode's samples
        independent of which episodes this agent ran before."""
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.history, self.queue = [], []
        self.diagnostic = {}
        self._previous_next_block = self._previous_prediction = None

    def _diagnostic_observation(self, z):
        """Compare only the previous one-block prediction with this real observation."""
        self.diagnostic = dict(active_goal_index=None, goal_changed=None, goal_distance_sq=None,
                               goal_threshold_sq=None, candidate_scores=None, selected_candidate=None,
                               plan_change_rms=None, prediction_mse=None)
        if self._previous_prediction is not None:
            self.diagnostic['prediction_mse'] = float((z-self._previous_prediction).square().mean())
        self._previous_prediction = None

    def _diagnostic_plan(self, blocks, best=0, prediction=None):
        # Raw, clipped action units, just like the actions returned to the evaluator.
        raw = np.clip(blocks[best, :2].detach().cpu().numpy()*self.a_std+self.a_mean, -1, 1)
        if self._previous_next_block is not None:
            self.diagnostic['plan_change_rms'] = float(np.sqrt(np.mean(
                np.square(raw[0]-self._previous_next_block))))
        self._previous_next_block = raw[1].copy() if len(raw) > 1 else None
        self._previous_prediction = prediction.detach().float().clone() if prediction is not None else None

    @torch.no_grad()
    def step(self, frame):
        x = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)[None, None]
        with self._autocast():
            z = self.encoder.encode(x)[0, 0].float()
        if self.diagnostics:
            self._diagnostic_observation(z)
        self.history = (self.history+[(z-self.z_mean)/self.z_std])[-self.policy.cfg.context:]
        if not self.queue:
            ctx = [self.history[0]]*(self.policy.cfg.context-len(self.history))+self.history
            samples = self.policy.sample(torch.stack(ctx)[None], self.generate, self.n_steps, self.lag,
                                         self.generator)
            if self.diagnostics:
                self._diagnostic_plan(samples)
            blocks = samples[0, :self.execute].cpu().numpy()
            raw = np.clip(blocks*self.a_std+self.a_mean, -1, 1).reshape(-1, STRIDE, RAW_ACTION_DIM)
            self.queue = list(raw)
        return self.queue.pop(0)


class GoalPlanningAgent(LatentPolicyAgent):
    """Test-time planner that scores candidate policy chunks with the latent world model.

    At each replanning step:
      1. Encode the current tiled RGB frame I_k into a latent z_k in R^192 and advance the
         subgoal index m whenever ||z_k - g_m||_2^2 < tau * ||g_m - g_{m-1}||_2^2.
      2. Draw `candidates` chunks of `generate` action blocks from the policy.
      3. Roll each candidate chunk forward in the world model from the last C<=3 real frame
         latents (and the action blocks executed between them) to predict latents z_hat.
      4. Execute the first `execute` blocks of the candidate whose final predicted latent
         z_hat_{k+generate} is closest in squared L2 distance to the active subgoal g_m.

    Goals are an ordered subgoal sequence; the index advances monotonically
    while ||z - g_m||_2^2 < tau * ||g_m - g_{m-1}||_2^2, with g_{-1} the latent of init_frame.
    Call reset(goal_frames, init_frame, seed) per episode and step(frame) every STRIDE steps.
    """

    def __init__(self, policy_path, device='cuda', candidates=16, tau=.25,
                 scoring='terminal', realignment=False, realignment_config=None,
                 realignment_mode='forward', adaptive_sampling=False, max_candidates=48,
                 min_improvement=.20, **kwargs):
        if scoring not in ('terminal', 'tail_min', 'tail_weighted'):
            raise ValueError('unknown candidate scoring mode')
        from assignment.training.planning.alignment import GoalRealigner
        if realignment_mode not in ('forward', 'bidirectional'):
            raise ValueError('unknown realignment search mode')
        self.scoring = scoring
        if adaptive_sampling and (scoring!='tail_min' or realignment or candidates!=16 or max_candidates!=48 or min_improvement!=.20):
            raise ValueError('G6 requires G1 scoring/goal selection and fixed 16/48/.20 sampling settings')
        self.adaptive_sampling, self.max_candidates = adaptive_sampling, max_candidates
        self.min_improvement = min_improvement
        self.realigner = GoalRealigner(realignment_config, bidirectional=realignment_mode=='bidirectional') if realignment else None
        super().__init__(policy_path, device, **kwargs)
        if scoring != 'terminal' and self.generate < 4:
            raise ValueError('tail scoring requires at least four generated blocks')
        if adaptive_sampling and (self.generate!=16 or self.execute!=1):
            raise ValueError('G6 requires generate=16, execute=1')
        self.candidates, self.tau = candidates, tau
        self.goals = None

    def reset(self, goal_frames=None, init_frame=None, seed=None):
        if goal_frames is not None and init_frame is None:
            raise ValueError('init_frame is required with goal_frames')
        super().reset(seed)
        self.frames, self.executed, self.index = [], [], 0
        self.last_distance = self.goals = self.gap = None
        if getattr(self, 'realigner', None) is not None:
            self.realigner.reset()
        if goal_frames is not None:
            self.goals = self._encode(np.asarray(goal_frames))
            prev = torch.cat([self._encode(np.asarray(init_frame)[None]), self.goals[:-1]])
            self.gap = (self.goals-prev).square().sum(-1)

    def goal_index(self, z):
        while self.index < len(self.goals)-1 and \
                (z-self.goals[self.index]).square().sum() < self.tau*self.gap[self.index]:
            self.index += 1
        return self.index

    def _select_real_goal(self, z):
        """Official arrival first; optional G3 matching uses only this real latent."""
        before = self.index
        self.goal_index(z)
        info = dict(goal_change_reason='official_reached' if self.index != before else None,
                    goal_change_from=before, goal_change_to=self.index,
                    goal_change_direction='forward' if self.index>before else 'unchanged')
        realigner = getattr(self, 'realigner', None)
        if realigner is None:
            return info
        distance = float((z-self.goals[self.index]).square().sum())
        def future():
            first = max(0, self.index-realigner.config.lookahead) if realigner.bidirectional else self.index+1
            return [dict(index=j, distance_sq=float((z-self.goals[j]).square().sum()),
                         threshold_sq=float(self.tau*self.gap[j]))
                    for j in range(first, min(self.index+realigner.config.lookahead+1, len(self.goals)))]
        matched, details = realigner.observe(self.index, distance, future)
        info.update(details)
        if matched is not None:
            origin = self.index
            self.index = matched
            if not realigner.bidirectional:
                self.goal_index(z)
            reason = ('realignment_backward' if matched < origin else 'realignment_forward') if realigner.bidirectional else ('realignment_then_official' if self.index != matched else 'realignment')
            info.update(goal_change_reason=reason,
                        alignment_match_direction='backward' if matched < origin else 'forward',
                        goal_change_to=self.index)
            realigner.changed(self.index, float((z-self.goals[self.index]).square().sum()))
        info['goal_change_direction'] = 'forward' if self.index>before else 'backward' if self.index<before else 'unchanged'
        return info

    def _candidate_cost(self, pred, goal):
        mode = getattr(self, 'scoring', 'terminal')
        if mode == 'terminal':
            # Preserve the original G0 reduction and argmin exactly.
            return (pred[:, -1].float()-goal).square().sum(-1)
        distances = (pred[:, -4:].float()-goal).square().sum(-1)
        if mode == 'tail_min':
            return distances.min(-1).values
        weights = distances.new_tensor([1., 2., 3., 4.])
        return (distances*weights).sum(-1)/10.

    def _adaptive_plan(self, pctx, goal, z):
        """Sample only new batches, keeping the same real context/goal and all earlier candidates."""
        from assignment.training.planning.sampling import improvement, stop_reason
        n = self.candidates
        context = torch.stack(pctx)[None].expand(n, -1, -1)
        frames = torch.stack(self.frames)[None].expand(n, -1, -1)
        past = self.executed[len(self.executed)-len(self.frames)+1:] if len(self.frames)>1 else []
        past_actions = torch.stack(past)[None].expand(n, -1, -1) if past else None
        d_now, threshold = float((z-goal).square().sum()), float(self.tau*self.gap[self.index])
        all_blocks, all_pred, all_costs, trace = [], [], [], []
        total, global_cost, global_index = 0, float('inf'), 0
        while True:
            blocks = self.policy.sample(context, self.generate, self.n_steps, self.lag, self.generator)
            acts = torch.cat([past_actions, blocks], 1) if past_actions is not None else blocks
            with self._autocast():
                pred = self.encoder.rollout(frames, self.encoder.action_encoder(acts), self.generate)
            costs = self._candidate_cost(pred, goal)
            local = int(costs.argmin())
            batch_cost = float(costs[local])
            previous = global_cost
            if batch_cost < global_cost:  # exact ties keep the earliest sampled candidate
                global_cost, global_index = batch_cost, total+local
            total += n
            reason = stop_reason(d_now, global_cost, threshold, total, self.max_candidates, self.min_improvement)
            trace.append(dict(batch=len(trace)+1, candidate_count=total, batch_best_cost=batch_cost,
                              cumulative_best_cost=global_cost, predicted_improvement=improvement(d_now, global_cost),
                              extra_batch_cost_gain=previous-global_cost if len(trace) else None,
                              best_candidate=global_index, resampled_next=reason is None, stop_reason=reason))
            all_blocks.append(blocks)
            all_pred.append(pred)
            all_costs.append(costs)
            if reason is not None:
                break
        self.last_distance = torch.cat(all_costs)
        info = dict(sampling_mode='adaptive', actual_candidates=total, sampling_batch_size=n,
                    max_candidates=self.max_candidates, min_improvement=self.min_improvement,
                    distance_now_sq=d_now, cumulative_best_costs=[r['cumulative_best_cost'] for r in trace],
                    batch_best_costs=[r['batch_best_cost'] for r in trace], sampling_batches=trace,
                    predicted_improvement=trace[-1]['predicted_improvement'], stop_reason=reason,
                    selected_batch=global_index//n+1, policy_sample_calls=len(trace), wm_rollout_calls=len(trace))
        return torch.cat(all_blocks), torch.cat(all_pred), global_index, info

    @torch.no_grad()
    def _encode(self, frames):
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)[:, None]
        with self._autocast():
            return self.encoder.encode(x)[:, 0].float()

    @torch.no_grad()
    def step(self, frame):
        if self.goals is None:
            raise RuntimeError('Call agent.reset(goal_frames, init_frame) before step().')
        z = self._encode(frame[None])[0]
        old_index = self.index
        if self.diagnostics:
            self._diagnostic_observation(z)
        ctx = self.encoder.cfg.context
        self.frames = (self.frames+[z])[-ctx:]
        self.history = (self.history+[(z-self.z_mean)/self.z_std])[-self.policy.cfg.context:]
        goal_info = self._select_real_goal(z)
        goal = self.goals[self.index]
        if self.diagnostics:
            self.diagnostic.update(active_goal_index=self.index, goal_changed=self.index != old_index,
                                   goal_distance_sq=float((z-goal).square().sum()),
                                   goal_threshold_sq=float(self.tau*self.gap[self.index]), **goal_info)
        if not self.queue:
            pctx = [self.history[0]]*(self.policy.cfg.context-len(self.history))+self.history
            n = self.candidates
            sampling_info = None
            if getattr(self, 'adaptive_sampling', False):
                blocks, pred, best, sampling_info = self._adaptive_plan(pctx, goal, z)
            else:
                blocks = self.policy.sample(torch.stack(pctx)[None].expand(n, -1, -1), self.generate, self.n_steps,
                                            self.lag, self.generator)
                best = 0
                pred = None
                if n > 1:
                    # Pair the C-1 earlier frames with executed blocks; current frame with new candidates.
                    past = self.executed[len(self.executed)-len(self.frames)+1:] if len(self.frames) > 1 else []
                    acts = torch.cat([torch.stack(past)[None].expand(n, -1, -1), blocks], 1) if past else blocks
                    with self._autocast():
                        pred = self.encoder.rollout(torch.stack(self.frames)[None].expand(n, -1, -1),
                                                    self.encoder.action_encoder(acts), self.generate)
                    self.last_distance = self._candidate_cost(pred, goal)
                    best = int(self.last_distance.argmin())
            if self.diagnostics:
                self.diagnostic.update(candidate_scores=self.last_distance.detach().cpu().tolist()
                                       if n > 1 else None, selected_candidate=best)
                self._diagnostic_plan(blocks, best, pred[best, 0] if pred is not None else None)
                self.diagnostic['scoring'] = getattr(self, 'scoring', 'terminal')
                if sampling_info is None and n>1:
                    from assignment.training.planning.sampling import improvement
                    cost = self.diagnostic['candidate_scores'][best]
                    sampling_info = dict(sampling_mode='fixed', actual_candidates=n, sampling_batch_size=n,
                                         max_candidates=n, min_improvement=None,
                                         distance_now_sq=self.diagnostic['goal_distance_sq'],
                                         cumulative_best_costs=[cost], batch_best_costs=[cost],
                                         sampling_batches=[dict(batch=1,candidate_count=n,batch_best_cost=cost,
                                                                cumulative_best_cost=cost,extra_batch_cost_gain=None)],
                                         predicted_improvement=improvement(self.diagnostic['goal_distance_sq'],cost),
                                         stop_reason='fixed_budget', selected_batch=1,
                                         policy_sample_calls=1, wm_rollout_calls=1)
                if sampling_info is not None:
                    self.diagnostic.update(sampling_info)
                if pred is not None:
                    selected_distances = (pred[best].float()-goal).square().sum(-1)
                    self.diagnostic.update(selected_goal_distances_sq=selected_distances.cpu().tolist(),
                                           selected_closest_block=int(selected_distances.argmin())+1,
                                           selected_tail_closest_block=int(selected_distances[-4:].argmin())+max(0, self.generate-4)+1)
            raw = np.clip(blocks[best, :self.execute].cpu().numpy()*self.a_std+self.a_mean, -1, 1)
            self.queue = list(raw.reshape(-1, STRIDE, RAW_ACTION_DIM))
        block = self.queue.pop(0)
        self.executed.append(torch.from_numpy((block.reshape(-1)-self.a_mean)/self.a_std).to(self.device))
        return block
