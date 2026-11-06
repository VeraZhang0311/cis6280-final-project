"""Check horizon selection, real context alignment and frozen-baseline equivalence."""
import contextlib
import types
import unittest
from unittest import mock

import numpy as np
import torch

from assignment.training.planning.agent import GoalPlanningAgent, LatentPolicyAgent


class Policy:
    cfg = types.SimpleNamespace(context=1, horizon=16)

    def __init__(self):
        self.calls = []

    def sample(self, context, generate, n_steps, lag, generator):
        self.calls.append((context.clone(), generate))
        blocks = torch.zeros(context.shape[0], generate, 125)
        # Candidate 0 reaches goal=1 at block 4 and then moves away.
        # Candidate 1 reaches it at block 16. Their first actions differ.
        blocks[0, :, 0] = .25
        blocks[1, 0, 0] = -.25
        blocks[1, 1:, 0] = 1.25 / 15
        return blocks


class Encoder:
    cfg = types.SimpleNamespace(context=3)

    def __init__(self):
        self.calls = []

    def action_encoder(self, acts):
        return acts

    def rollout(self, frames, acts, steps):
        self.calls.append((frames.clone(), acts.clone(), steps))
        prefix = frames.shape[1] - 1
        return frames[:, -1:] + acts[:, prefix:prefix + steps, :1].cumsum(1)


def setup_agent(agent):
    agent.device = 'cpu'
    agent._device_type = 'cpu'
    agent._autocast = contextlib.nullcontext
    agent.generator = torch.Generator().manual_seed(0)
    agent.policy, agent.encoder = Policy(), Encoder()
    agent.generate, agent.execute, agent.n_steps, agent.lag = 16, 1, 10, 0
    agent.z_mean, agent.z_std = torch.zeros(1), torch.ones(1)
    agent.a_mean, agent.a_std = np.zeros(125, np.float32), np.ones(125, np.float32)
    agent._encode = lambda frames: torch.from_numpy(np.asarray(frames)).float().reshape(len(frames), -1)
    LatentPolicyAgent.reset(agent)


def fake_base_init(self, *args, **kwargs):
    setup_agent(self)


def make_agent(blocks=None, cls=GoalPlanningAgent):
    with mock.patch.object(LatentPolicyAgent, '__init__', fake_base_init):
        if cls is GoalPlanningAgent:
            agent = cls('unused', candidates=2, planning_horizon=blocks)
        else:
            agent = cls('unused', candidates=2)
    agent.reset(np.array([[1]], np.uint8), np.array([0], np.uint8), seed=100038)
    return agent


class HorizonTests(unittest.TestCase):
    def test_requested_prefix_only_and_full_policy_generation(self):
        for blocks in (4, 8, 16):
            agent = make_agent(blocks)
            for _ in range(4):
                action = agent.step(np.array([0], np.uint8))
                self.assertEqual(action.shape, (5, 25))
                self.assertEqual(action.dtype, np.float32)
                self.assertTrue(np.isfinite(action).all())
                self.assertLessEqual(float(np.abs(action).max()), 1)
            self.assertTrue(all(generate == 16 for _, generate in agent.policy.calls))
            for frames, acts, steps in agent.encoder.calls:
                self.assertEqual(steps, blocks)
                self.assertEqual(acts.shape[1], frames.shape[1] - 1 + blocks)
            # Context must contain actual previously executed actions.
            torch.testing.assert_close(agent.encoder.calls[1][1][0, 0], agent.executed[0])
            torch.testing.assert_close(agent.encoder.calls[2][1][0, :2], torch.stack(agent.executed[:2]))

    def test_horizon_changes_candidate_selection(self):
        short, long = make_agent(4), make_agent(16)
        self.assertGreater(float(short.step(np.array([0], np.uint8))[0, 0]), 0)
        self.assertLess(float(long.step(np.array([0], np.uint8))[0, 0]), 0)

    def test_default_matches_explicit_eight_seconds(self):
        default, explicit = make_agent(), make_agent(16)
        for _ in range(5):
            np.testing.assert_array_equal(default.step(np.array([0], np.uint8)),
                                          explicit.step(np.array([0], np.uint8)))
            torch.testing.assert_close(default.last_distance, explicit.last_distance)

    def test_invalid_horizons(self):
        for blocks in (0, -1, 17, 4.5, True):
            with self.assertRaises(ValueError):
                make_agent(blocks)

    def test_real_observations_alone_advance_goals(self):
        agent = make_agent(4)
        agent.reset(np.array([[1], [2]], np.uint8), np.array([0], np.uint8))
        agent.step(np.array([0], np.uint8))
        self.assertEqual(agent.index, 0)  # Imagined candidate reaches goal 1.
        agent.step(np.array([1], np.uint8))
        self.assertEqual(agent.index, 1)


if __name__ == '__main__':
    unittest.main()
