"""Planner decisions on controlled latent trajectories, without downloaded models."""
import contextlib
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from assignment.training.planning.agent import GoalPlanningAgent, TrajectoryPlanningAgent


def planner(**kwargs):
    agent = TrajectoryPlanningAgent.__new__(TrajectoryPlanningAgent)
    agent.terminal_weight = kwargs.get('terminal_weight', .25)
    agent.lookahead_weight = kwargs.get('lookahead_weight', .15)
    agent.time_weight = .05
    agent.adaptive, agent.patience = True, 3
    agent.tau = .25
    agent.generator = torch.Generator().manual_seed(0)
    agent._encode = lambda x: torch.as_tensor(x, dtype=torch.float32).flatten(1)
    agent.reset(np.array([[1.], [2.]], np.float32), np.array([0.], np.float32), 123)
    return agent


class PlanningTests(unittest.TestCase):
    def test_worker_respects_allocated_gpu_mask(self):
        from assignment.training.planning.evaluate import init_worker
        counter = SimpleNamespace(value=0, get_lock=contextlib.nullcontext)
        with mock.patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '3,5'}):
            init_worker(counter, [1])
            import os
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '5')
            self.assertEqual(os.environ['MUJOCO_EGL_DEVICE_ID'], '5')

    def test_trajectory_recognizes_intermediate_arrival(self):
        agent = planner()
        pred = torch.tensor([[[1.], [1.8]], [[1.5], [1.5]]])
        goal = agent.goals[0]
        self.assertEqual(int(GoalPlanningAgent.score_candidates(agent, pred, goal).argmin()), 1)
        self.assertEqual(int(agent.score_candidates(pred, goal).argmin()), 0)

    def test_next_goal_bonus_requires_ordered_arrival(self):
        agent = planner(terminal_weight=0.)
        pred = torch.tensor([[[2.], [1.]], [[1.], [2.]]])
        score = agent.score_candidates(pred, agent.goals[0])
        self.assertLess(float(score[1]), float(score[0]))
        # No next-goal reward when current goal is never reached.
        pred = torch.tensor([[[1.6], [2.]]])
        with_bonus = agent.score_candidates(pred, agent.goals[0])
        agent.lookahead_weight = 0.
        torch.testing.assert_close(with_bonus, agent.score_candidates(pred, agent.goals[0]))

    def test_last_goal_and_zero_gap_stay_finite(self):
        agent = planner()
        agent.index = 1
        agent.gap[1] = 0
        self.assertTrue(torch.isfinite(agent.score_candidates(torch.ones(2, 3, 1), agent.goals[1])).all())

    def test_stall_does_not_skip_an_unreached_goal(self):
        agent = planner()
        for _ in range(20):
            self.assertEqual(agent.goal_index(torch.tensor([0.])), 0)
        # Past the old goal, near the next one, and no further improvement.
        for _ in range(4):
            agent.goal_index(torch.tensor([1.6]))
        self.assertEqual(agent.index, 1)
        self.assertIn('passed', agent.switches)
        agent.reset()
        self.assertEqual(agent.stale, 0)
        self.assertEqual(agent.switches, [])

    def test_predicts_clipped_executable_actions(self):
        agent = planner()
        agent.a_mean = np.zeros(125, np.float32)
        agent.a_std = np.full(125, 2., np.float32)
        blocks = torch.full((2, 3, 125), 4.)
        clipped = agent.prepare_candidates(blocks)
        torch.testing.assert_close(clipped, torch.full_like(blocks, .5))

    def test_closed_loop_uses_history_and_returns_valid_blocks(self):
        agent = planner()
        agent.device = 'cpu'
        agent.candidates, agent.generate, agent.execute = 2, 3, 2
        agent.n_steps, agent.lag = 1, 0
        agent.a_mean = np.zeros(125, np.float32)
        agent.a_std = np.ones(125, np.float32)
        agent.z_mean, agent.z_std = torch.zeros(1), torch.ones(1)
        calls = []

        def rollout(frames, actions, steps):
            calls.append((frames.shape[1], actions.shape[1]))
            self.assertEqual(actions.shape[1], frames.shape[1]+steps-1)
            return torch.zeros(2, steps, 1)

        agent.encoder = SimpleNamespace(cfg=SimpleNamespace(context=3), action_encoder=lambda x: x, rollout=rollout)
        agent.policy = SimpleNamespace(cfg=SimpleNamespace(context=1),
                                       sample=lambda *args: torch.zeros(2, 3, 125))
        agent._autocast = contextlib.nullcontext
        for _ in range(5):
            block = agent.step(np.array([0.], np.float32))
            self.assertEqual(block.shape, (5, 25))
            self.assertEqual(block.dtype, np.float32)
            self.assertTrue((np.abs(block) <= 1).all())
        self.assertEqual(calls, [(1, 3), (3, 5), (3, 5)])


if __name__ == '__main__':
    unittest.main()
