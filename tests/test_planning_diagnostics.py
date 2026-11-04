"""Alignment and behavior-preservation checks; no checkpoints or GPU required."""
import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from assignment.training.planning.agent import GoalPlanningAgent, LatentPolicyAgent
from assignment.training.planning.diagnostics import EpisodeDiagnostics, aggregate, analyze_run
from assignment.training.planning.phase1 import init_phase1_worker, validate_goals


class FakePolicy:
    cfg = SimpleNamespace(context=1, horizon=16)

    def sample(self, ctx, generate, n_steps, lag, generator):
        return torch.randn((len(ctx), generate, 125), generator=generator)*.2


class FakeEncoder:
    cfg = SimpleNamespace(context=3)

    def __init__(self):
        self.rollout_calls = 0

    def encode(self, x):
        return x.float().flatten(2).mean(-1, keepdim=True).expand(-1, -1, 2)/255

    def action_encoder(self, acts):
        return acts

    def rollout(self, frames, acts, steps):
        self.rollout_calls += 1
        return frames[:, -1:] + acts[:, -steps:, :2].cumsum(1)


def fake_agent(cls, diagnostics):
    agent = cls.__new__(cls)
    agent.encoder, agent.policy = FakeEncoder(), FakePolicy()
    agent.device, agent.diagnostics = 'cpu', diagnostics
    agent._autocast = contextlib.nullcontext
    agent.z_mean, agent.z_std = torch.zeros(2), torch.ones(2)
    agent.a_mean, agent.a_std = np.zeros(125, np.float32), np.ones(125, np.float32)
    agent.generate, agent.execute, agent.n_steps, agent.lag = 16, 1, 10, 0
    agent.generator = torch.Generator()
    if cls is GoalPlanningAgent:
        agent.candidates, agent.tau = 16, .25
        agent.reset(np.full((3, 2, 2, 3), [20, 100, 200], np.uint8),
                    np.zeros((2, 2, 3), np.uint8), seed=37)
    else:
        agent.reset(seed=37)
    return agent


class AgentDiagnostics(unittest.TestCase):
    def test_logging_preserves_actions_rng_and_rollout_count(self):
        for cls in (LatentPolicyAgent, GoalPlanningAgent):
            plain, logged = fake_agent(cls, False), fake_agent(cls, True)
            for value in (0, 40, 80, 120):
                frame = np.full((2, 2, 3), value, np.uint8)
                np.testing.assert_array_equal(plain.step(frame), logged.step(frame))
                self.assertTrue(torch.equal(plain.generator.get_state(), logged.generator.get_state()))
                self.assertEqual(plain.encoder.rollout_calls, logged.encoder.rollout_calls)
            if cls is LatentPolicyAgent:
                self.assertEqual(logged.encoder.rollout_calls, 0)
                self.assertIsNone(logged.diagnostic['prediction_mse'])
                self.assertIsNone(logged.diagnostic['candidate_scores'])
                self.assertIsNone(logged.diagnostic['active_goal_index'])

    def test_prediction_and_plan_are_aligned_with_previous_plan(self):
        agent = fake_agent(GoalPlanningAgent, True)
        agent.step(np.zeros((2, 2, 3), np.uint8))
        self.assertIsNone(agent.diagnostic['prediction_mse'])
        self.assertIsNone(agent.diagnostic['plan_change_rms'])
        expected_next = agent._previous_next_block.copy()
        prediction = agent._previous_prediction.clone()
        frame = np.full((2, 2, 3), 40, np.uint8)
        z = agent._encode(frame[None])[0]
        block = agent.step(frame)
        self.assertAlmostEqual(agent.diagnostic['prediction_mse'], float((z-prediction).square().mean()))
        self.assertAlmostEqual(agent.diagnostic['plan_change_rms'],
                               float(np.sqrt(np.mean((block.reshape(-1)-expected_next)**2))))
        agent.reset(np.zeros((1, 2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8), seed=37)
        self.assertIsNone(agent._previous_prediction)
        self.assertIsNone(agent._previous_next_block)

    def test_goal_diagnostics_use_post_switch_goal_including_skips(self):
        agent = fake_agent(GoalPlanningAgent, True)
        agent.goals = torch.tensor([[0., 0.], [0., 0.], [2., 2.]])
        agent.gap = torch.ones(3)
        agent.step(np.zeros((2, 2, 3), np.uint8))
        self.assertEqual(agent.diagnostic['active_goal_index'], 2)
        self.assertTrue(agent.diagnostic['goal_changed'])
        self.assertEqual(agent.diagnostic['goal_distance_sq'], 8.)
        self.assertEqual(agent.diagnostic['goal_threshold_sq'], .25)


class TaskDiagnostics(unittest.TestCase):
    def test_slurm_worker_selects_within_allocation(self):
        with mock.patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '3,6'}), \
                mock.patch('assignment.training.planning.phase1.init_worker') as init:
            init_phase1_worker('counter', [0, 1])
        init.assert_called_once_with('counter', [3, 6])

    def test_evaluator_records_control_events_and_roundtrips_outputs(self):
        from assignment.training.planning import evaluate
        agent = fake_agent(LatentPolicyAgent, True)
        agent.world_model_path = '/fake/model.pt'
        class Env:
            def __init__(self, *args):
                self.env = SimpleNamespace(render=lambda: [np.zeros((128, 128, 3), np.uint8)]*2,
                                           history=[{}])
                self.members = np.zeros(4, bool)
                self.n = 0
            def reset(self, *args):
                pass
            def step(self, action):
                self.n += 1
                count = 1 if self.n == 2 else 0
                return None, 0., False, self.n == 10, dict(placed=count, count=count, success=False, ground=0)
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(evaluate, 'make_agent', return_value=agent), \
                mock.patch.object(evaluate, 'runtime_information', return_value={}), \
                mock.patch('torch.cuda.synchronize'), \
                mock.patch('assignment.training.data.task.StackTaskEnv', Env), \
                mock.patch('assignment.training.data.signals.row_hand_parts', return_value=(np.zeros(4), None)), \
                mock.patch('assignment.training.data.signals.row_lifted', return_value=np.zeros(4, bool)):
            folder = Path(root)/'B0/episodes/scene-100038.sample-0'
            result = evaluate.run_episode(('df', 'fake', 'fake', dict(execute=1), 100038,
                                           dict(horizon=10), 128, str(folder/'video.mp4'), None, 0, folder,
                                           dict(tail_state_steps=0, keyframes=2)))
            self.assertEqual(result['stage_steps'], [2])
            self.assertEqual(result['first_placement_step'], 2)
            self.assertEqual(result['max_placed'], 1)
            self.assertEqual(result['final_count'], 0)
            rows = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['control_step'] for r in rows], [0, 5])
            self.assertEqual([r['stacked_count'] for r in rows], [0, 0])
            self.assertEqual(rows[1]['max_stacked_count'], 1)
            self.assertTrue((folder/'timeline.png').stat().st_size > 1000)
            self.assertTrue((folder/'video.mp4').stat().st_size > 1000)
            import imageio.v2 as imageio
            reader = imageio.get_reader(folder/'video.mp4')
            try:
                self.assertEqual(reader.count_frames(), 3)  # t=0, .5, and terminal t=1
            finally:
                reader.close()
            self.assertEqual(len(list((folder/'keyframes').glob('*.png'))), 2)
            self.assertTrue((folder/'keyframes/step-10.png').exists())
            summary = analyze_run(root)
            self.assertEqual(summary['B0']['reached']['placed>=1'], 1.)
            self.assertIsNone(json.loads((Path(root)/'evidence.json').read_text())[0]['max_goal_dwell_s'])
            review = Path(root)/'review.csv'
            review.write_text('manual annotation', encoding='utf-8')
            analyze_run(root)
            self.assertEqual(review.read_text(), 'manual annotation')

    def test_tail_evidence_is_bounded_copies_rows_and_records_hand_exclusion(self):
        from assignment.training.planning.evidence import TailEvidence
        row = dict(object_bounds=np.array([[[.13, -.02, 0.], [.18, .02, .05]]]),
                   object_hand_contacts=np.zeros(1, int), object_ground_contacts=np.zeros(1, int),
                   object_table_support=np.ones(1, bool), object_support=np.zeros((1, 1), bool))
        raw = SimpleNamespace(history=[row], object_geoms=[], data=SimpleNamespace(contact=[]),
                              config=SimpleNamespace(objects=[SimpleNamespace(material='orange', size=(.05,)*3)]))
        env = SimpleNamespace(env=raw, goal=(.16, 0.), half=.085)
        tail = TailEvidence(2)
        tail.capture(env, 1)
        tail.capture(env, 2)
        row['object_hand_contacts'][0] = 1
        tail.capture(env, 3)
        self.assertEqual([r['control_step'] for r in tail.rows], [2, 3])
        self.assertEqual([r['count'] for r in tail.rows], [1, 0])
        self.assertEqual(tail.rows[0]['object_hand_contacts'], [0])
        with tempfile.TemporaryDirectory() as root:
            tail.save(root, env)
            self.assertEqual(len(json.loads((Path(root)/'tail-state.json').read_text())), 2)
            self.assertEqual(json.loads((Path(root)/'objects.json').read_text())[0]['material'], 'orange')

    def test_events_keep_between_block_transients_and_censoring(self):
        recorder = EpisodeDiagnostics()
        recorder.task_step(2, 1)
        recorder.task_step(3, 0)
        self.assertEqual([e['event'] for e in recorder.events], ['first_box', 'stack_count_drop'])
        # A short-lived first box must not be lost even if both render points show zero.
        recorder.steps = [dict(decision_s=.2)]
        result = dict(steps=10, success=False, final_count=0, max_placed=1)
        result.update(recorder.finish(result))
        self.assertIsNone(result['first_to_second_s'])
        self.assertFalse(result['second_reached'])
        self.assertEqual(result['post_first_observed_s'], .8)

    def test_summary_does_not_turn_censored_time_into_transition(self):
        base = dict(success=False, steps=10, grasp_steps=0, lifted_steps=0, ground_falls=0,
                    final_count=0, first_placement_step=2, post_first_observed_s=.8)
        rows = [dict(base, max_placed=1, second_reached=False, first_to_second_s=None),
                dict(base, max_placed=2, second_reached=True, first_to_second_s=.3)]
        summary = aggregate(rows, [.1, .3])
        self.assertEqual(summary['second_given_first'], .5)
        self.assertEqual(summary['first_to_second_s_mean'], .3)
        self.assertEqual(summary['post_first_censored_episodes'], 1)
        self.assertAlmostEqual(summary['decision_s_p95'], .29)

    def test_official_final_off_grid_goal_is_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'goals.npz'
            def save(steps):
                np.savez(path, seeds=[100038], sub_count=[len(steps)], sub_steps=[steps],
                         init_frames=np.zeros((1, 2, 2, 3), np.uint8),
                         sub_frames=np.zeros((1, len(steps), 2, 2, 3), np.uint8))
            save([80, 160, 199])
            validate_goals(path, [100038])
            save([40, 80, 120])
            with self.assertRaises(ValueError):
                validate_goals(path, [100038])


if __name__ == '__main__':
    unittest.main()
