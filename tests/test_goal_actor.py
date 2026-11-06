"""v2 selection, real-observation goal updates, action alignment, and evidence."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch is not installed')
class SelectionTests(unittest.TestCase):
    def test_a2_uses_all_future_blocks_and_retains_first_tie(self):
        from assignment.training.planning.goal_actor import select_candidate
        distances = torch.full((3, 16), 10.)
        distances[0, 0], distances[1, -1], distances[2, 4] = .3, .2, .2
        result = select_candidate(distances, 'A2', 1)
        self.assertEqual(result.best, 1)
        self.assertEqual(result.nearest_block.tolist(), [1, 16, 5])
        with self.assertRaises(ValueError):
            select_candidate(torch.zeros(3, 17), 'A2', 1)  # no current observation in min

    def test_a3_normalizes_by_each_threshold_and_can_choose_larger_active_cost(self):
        from assignment.training.planning.goal_actor import select_candidate
        distances = torch.full((3, 16), 100.)
        distances[0, 0], distances[1, 0], distances[2, 0] = .4, 1., 2.
        result = select_candidate(distances, 'A3', 2., torch.tensor([32., 0., 0.]), 8.)
        torch.testing.assert_close(result.q, torch.tensor([.2, .5, 1.]))
        torch.testing.assert_close(result.r, torch.tensor([4., 0., 0.]))
        self.assertEqual(result.eligible.tolist(), [True, True, False])
        self.assertTrue(result.lookahead_used)
        self.assertEqual(result.best, 1)
        self.assertGreater(float(result.active_min[1]), float(result.active_min[0]))
        self.assertIsNone(result.trace()['selection_cost'][2])
        json.dumps(result.trace(), allow_nan=False)

    def test_a3_excludes_block_16_for_eligibility_and_falls_back_over_all_16(self):
        from assignment.training.planning.goal_actor import select_candidate
        distances = torch.full((2, 16), 2.)
        distances[0, -1] = .1
        result = select_candidate(distances, 'A3', 1., torch.zeros(2), 1.)
        self.assertEqual(result.fallback, 'no_eligible')
        self.assertEqual(result.best, 0)
        self.assertEqual(result.nearest_block[0], 16)
        distances[1, 0] = .9
        result = select_candidate(distances, 'A3', 1., torch.tensor([0., 100.]), 1.)
        self.assertEqual(result.eligible.tolist(), [False, True])
        self.assertEqual(result.best, 1)
        last = select_candidate(distances, 'A3', 1.)
        self.assertEqual(last.fallback, 'last_goal')
        self.assertEqual(last.best, 0)

    def test_zero_threshold_epsilon_and_lookahead_ties(self):
        from assignment.training.planning.goal_actor import select_candidate
        distances = torch.full((2, 16), 1e-9)
        result = select_candidate(distances, 'A3', 0., torch.zeros(2), 0.)
        self.assertEqual(result.best, 0)
        self.assertTrue(torch.isfinite(result.cost).all())
        self.assertTrue(result.eligible.all())
        distances[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            select_candidate(distances, 'A2', 1.)

    def test_trajectory_scores_match_predeclared_weights_and_stable_ties(self):
        from assignment.training.planning.goal_actor import select_candidate
        distances = torch.full((3, 16), 4.)
        # Candidate 0 arrives earliest but does not retain the goal. Candidate 1
        # arrives later and stays. Candidate 2 ties candidate 1 exactly.
        distances[0, 0], distances[0, -1] = .1, .8
        distances[1, 7], distances[1, -1] = .2, .2
        distances[2] = distances[1]
        t1 = select_candidate(distances, 'A4-32-T1', 1.)
        t2 = select_candidate(distances, 'A4-32-T2', 1.)
        self.assertEqual(t1.best, 0)
        self.assertEqual(t2.best, 1)
        self.assertEqual(t1.arrival.tolist(), [1, 8, 8])
        self.assertEqual(t1.nearest_block.tolist(), [1, 8, 8])
        self.assertEqual(t1.selection_rule, 'trajectory_arrival_terminal')
        self.assertEqual(t1.arrival_time_weight, .05)
        self.assertEqual(t1.terminal_weight, .25)
        self.assertEqual(t2.terminal_weight, .5)
        self.assertEqual(t2.best, 1)  # Candidate 1 retains the earlier exact tie.
        json.dumps(t2.trace(), allow_nan=False)

    def test_trajectory_score_uses_reference_gap_normalization(self):
        from assignment.training.planning.goal_actor import select_candidate
        distances = torch.ones(1, 16)
        distances[0, 3], distances[0, -1] = .25, .5
        result = select_candidate(distances, 'A4-32-T1', .5)
        # threshold=.5 and tau=.25 imply reference gap=2.
        expected = .25/2 + .05*4/16 + .25*.5/2
        self.assertAlmostEqual(float(result.cost[0]), expected, places=6)


@unittest.skipIf(torch is None, 'torch is not installed')
class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wm_path = Path(self.temp.name)/'wm.pt'
        self.wm_path.write_bytes(b'mocked weights')
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.cfg = types.SimpleNamespace(latent_dim=2, action_dim=125, context=1, horizon=16)
                self.received = []
            def sample(self, current, goal, *, steps, n_steps, lag, generator):
                assert (steps, n_steps, lag) == (16, 10, 0)
                self.received.append((current.clone(), goal.clone()))
                marker = len(self.received)/10
                blocks = torch.full((len(current), 16, 125), marker)
                blocks += torch.arange(16)[None, :, None]/1000
                blocks[:, :, 0] = 2  # Must be clipped *before* scoring and storing history.
                return blocks
        class ActionEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.received = []
            def forward(self, actions):
                self.received.append(actions.clone())
                return actions
        class WM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.cfg = types.SimpleNamespace(dim=2, action_dim=125, context=3, proprio_dim=0)
                self.action_encoder = ActionEncoder()
                self.rollouts = []
            def encode(self, frames):
                red = frames[:, :, 0, 0, 0].float()
                return torch.stack([red, torch.zeros_like(red)], -1)
            def rollout(self, context, actions, steps):
                self.rollouts.append((context.clone(), actions.clone(), steps))
                prediction = torch.zeros(len(context), steps, 2)
                prediction[..., 0] = 30  # Imagined goal reached; must not advance real active index.
                return prediction
        self.policy, self.wm = Policy(), WM()
        info = dict(encoder=str(self.wm_path), z_mean=[1., 0.], z_std=[2., 1.],
                    action_mean=[0.]*125, action_std=[1.]*125)
        from assignment.training.training.goal_policy import A4_PARENT_SHA256
        info.update(cumulative_step=150000, step=50000, prior_optimizer_steps=100000,
                    data=dict(include_tail=True), init_policy=dict(sha256=A4_PARENT_SHA256),
                    checkpoint_selection=dict(rule='final_step_ema'))
        self.info = info
        patches = [mock.patch('assignment.training.planning.goal_actor.load_goal_policy', return_value=(self.policy, info)),
                   mock.patch('assignment.training.planning.goal_actor.load_model',
                              return_value=(self.wm, dict(action_mean=[0.]*25, action_std=[1.]*25)))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def frame(value):
        return np.full((1, 1, 3), value, np.uint8)

    def agent(self, mode):
        from assignment.training.planning.goal_actor import GoalActorAgent
        agent = GoalActorAgent('goal.pt', 'cpu', mode=mode, world_model=self.wm_path)
        agent.reset(np.stack([self.frame(30), self.frame(60)]), self.frame(0), seed=4)
        return agent

    def test_a1_never_uses_wm_predictor_and_replans_each_block(self):
        agent = self.agent('A1')
        with mock.patch.object(self.wm, 'rollout', side_effect=AssertionError('A1 predictor')):
            first = agent.step(self.frame(0))
            second = agent.step(self.frame(0))
        self.assertEqual(first.shape, (5, 25))
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(first[0, 0], 1)
        self.assertAlmostEqual(first[0, 1], .1)
        self.assertAlmostEqual(second[0, 1], .2)  # new first block, not previous second block (.101)
        self.assertEqual(agent.total_calls['predictor'], 0)
        self.assertEqual(agent.total_calls['policy'], 2)
        self.assertEqual(len(self.wm.action_encoder.received), 0)
        self.assertEqual(agent.last_trace['candidates'], 1)
        current, goal = self.policy.received[0]
        torch.testing.assert_close(current, torch.tensor([[[-.5, 0.]]]))
        torch.testing.assert_close(goal, torch.tensor([[[14.5, 0.]]]))
        self.assertFalse(self.policy.training or self.wm.training)
        self.assertFalse(any(p.requires_grad for p in self.policy.parameters()))

    def test_a2_context_aligns_actual_clipped_actions_and_stays_bounded(self):
        agent = self.agent('A2')
        for red in (0, 1, 2, 3):
            agent.step(self.frame(red))
        self.assertEqual([record[0].shape[1] for record in self.wm.rollouts], [1, 2, 3, 3])
        self.assertEqual([record[1].shape[1] for record in self.wm.rollouts], [16, 17, 18, 18])
        context, actions, steps = self.wm.rollouts[-1]
        self.assertEqual(context[0, :, 0].tolist(), [1, 2, 3])
        self.assertAlmostEqual(float(actions[0, 0, 1]), .2)
        self.assertAlmostEqual(float(actions[0, 1, 1]), .3)
        self.assertAlmostEqual(float(actions[0, 2, 1]), .4)
        self.assertTrue((actions[:, :, 0] == 1).all())
        self.assertEqual(steps, 16)
        self.assertEqual(len(agent.executed), 2)
        self.assertEqual(agent.total_calls['predictor'], 64)

    def test_a3_only_real_observation_advances_goal_and_actor_never_sees_next(self):
        agent = self.agent('A3')
        agent.step(self.frame(0))
        self.assertEqual(agent.index, 0)
        self.assertTrue(agent.last_trace['lookahead_used'])
        self.assertEqual(agent.last_trace['calls']['wm_rollout'], 1)
        self.assertEqual(agent.last_trace['calls']['predictor'], 16)
        torch.testing.assert_close(self.policy.received[0][1], torch.tensor([[[14.5, 0.]]]).expand(16, 1, 2))
        agent.step(self.frame(30))
        self.assertEqual(agent.index, 1)
        self.assertEqual(len(agent.last_trace['switches']), 1)
        self.assertEqual(agent.last_trace['fallback'], 'last_goal')
        self.assertEqual(agent.thresholds.tolist(), [225., 225.])
        torch.testing.assert_close(self.policy.received[1][1], torch.tensor([[[29.5, 0.]]]).expand(16, 1, 2))
        json.dumps(agent.last_trace, allow_nan=False)

    def test_reset_validation_reseeding_and_strict_zero_threshold(self):
        agent = self.agent('A1')
        for burn in (0, 5):
            torch.rand(burn, generator=agent.generator)
            agent.reset(np.stack([self.frame(0), self.frame(0)]), self.frame(0), seed=10)
            draw = torch.rand(4, generator=agent.generator)
            if burn == 0:
                reference = draw
            else:
                self.assertTrue(torch.equal(reference, draw))
        agent.step(self.frame(0))
        self.assertEqual(agent.index, 0)
        with self.assertRaises(ValueError):
            agent.reset(np.stack([self.frame(30)]))
        self.assertEqual(len(agent.goals), 2)
        agent.reset()
        with self.assertRaises(RuntimeError):
            agent.step(self.frame(0))

    def test_a4_modes_use_expected_scoring_and_candidate_budget(self):
        for mode, count, rule in (('A4-16', 16, 'active_min'), ('A4-32', 32, 'active_min'),
                                  ('A4-32-T1', 32, 'trajectory_arrival_terminal'),
                                  ('A4-32-T2', 32, 'trajectory_arrival_terminal')):
            agent = self.agent(mode)
            block = agent.step(self.frame(0))
            self.assertEqual(block.shape, (5, 25))
            self.assertEqual(self.policy.received[-1][0].shape, (count, 1, 2))
            self.assertEqual(self.wm.rollouts[-1][0].shape[0], count)
            self.assertEqual(agent.last_trace['candidates'], count)
            self.assertEqual(agent.last_trace['selection_rule'], rule)
            self.assertFalse(agent.last_trace['lookahead_used'])
            self.assertEqual(agent.last_trace['calls']['predictor'], 16)
            if mode.endswith(('T1', 'T2')):
                self.assertEqual(len(agent.last_trace['candidate_arrival_block']), 32)
                self.assertEqual(len(agent.last_trace['candidate_terminal_distance']), 32)
            self.assertEqual(len(agent.last_trace['candidate_min_distance']), count)
        self.info['cumulative_step'] = 100000
        with self.assertRaisesRegex(ValueError, '150k'):
            self.agent('A4-32')

    def test_intermediate_a4_requires_explicit_matching_step_and_tail_provenance(self):
        from assignment.training.planning.goal_actor import GoalActorAgent
        from assignment.training.training.goal_policy import validate_a4_checkpoint
        self.info.update(cumulative_step=120000, step=20000)
        with self.assertRaisesRegex(ValueError, '150k'):
            self.agent('A4-16')
        for mode in ('A4-16', 'A4-32'):
            agent = GoalActorAgent('goal.pt', 'cpu', mode=mode, world_model=self.wm_path,
                                   a4_cumulative_step=120000)
            agent.reset(np.stack([self.frame(30), self.frame(60)]), self.frame(0), seed=4)
            self.assertEqual(agent.step(self.frame(0)).shape, (5, 25))
        with self.assertRaises(ValueError):
            validate_a4_checkpoint(self.info, 125000)
        self.info['data']['include_tail'] = False
        with self.assertRaises(ValueError):
            validate_a4_checkpoint(self.info, 120000)


class ReportingTests(unittest.TestCase):
    def test_resume_reuses_only_complete_matching_episodes(self):
        from assignment.training.planning.evaluate_goal_actor import completed_episode, run_episode
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            job = dict(folder=directory, mode='A1', scene=100038, sample_seed=2, horizon=20, video=True, resume=True)
            self.assertIsNone(completed_episode(job))  # partial episode must be rerun
            result = dict(mode='A1', scene_seed=100038, sample_seed=2, policy_seed=2100038,
                          steps=20, video='video.mp4')
            for name in ('steps.jsonl', 'counts.jsonl', 'video.mp4'):
                (folder/name).write_bytes(b'completed evidence')
            (folder/'episode.json').write_text(json.dumps(result))
            with mock.patch('assignment.training.planning.evaluate_goal_actor.make_agent',
                            side_effect=AssertionError('Cached episodes must not load a model')):
                reused = run_episode(job)
            self.assertTrue(reused.pop('_reused'))
            self.assertEqual(reused, result)
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                completed_episode(dict(job, sample_seed=0))
            (folder/'video.mp4').write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                completed_episode(job)

    def test_resume_rejects_changed_checkpoint_configuration_or_source(self):
        from assignment.training.planning.evaluate_goal_actor import validate_resume
        keys = ('groups', 'scene_seeds', 'sample_seeds', 'split', 'policy_seed', 'models', 'actor_config',
                'actor_interface', 'training_source', 'config', 'task', 'runtime')
        original = {key: 'unchanged' for key in keys}
        original['code'] = dict(source_sha256={'agent.py': 'abc'})
        validate_resume(original, original)
        for key in ('models', 'config', 'scene_seeds', 'runtime'):
            with self.assertRaises(ValueError):
                validate_resume(original, dict(original, **{key: 'changed'}))
        with self.assertRaisesRegex(ValueError, 'source snapshot'):
            validate_resume(original, dict(original, code=dict(source_sha256={'agent.py': 'def'})))

    def test_censoring_and_scene_grouped_pairs(self):
        from assignment.training.planning.evaluate_goal_actor import km_median, paired_comparisons, summarize
        def episode(scene, sample, peak, event):
            return dict(scene_seed=scene, sample_seed=sample, peak_boxes=peak, final_boxes=1, success=False,
                        decision_times_s=[.1, .3], calls=dict(predictor=0), decisions=2,
                        lookahead_applicable=0, eligible_nonempty=0, lookahead_used=0,
                        first_to_second=dict(at_risk=True, event=event, duration_s=10. if event else 20.))
        a = [episode(1, 0, 2, True), episode(1, 2, 2, True), episode(2, 0, 1, False), episode(2, 2, 1, False)]
        b = [episode(r['scene_seed'], r['sample_seed'], 1, False) for r in a]
        summary = summarize(a)
        self.assertEqual(summary['first_to_second']['censored'], 2)
        self.assertEqual(summary['first_to_second']['conversion'], .5)
        self.assertEqual(summary['first_to_second']['km_median_s'], 10.)
        self.assertEqual(summary['peak_reached']['2'], .5)
        self.assertIsNone(km_median([dict(at_risk=True, event=False, duration_s=20.)]))
        paired = paired_comparisons(dict(A1=b, A2=a))['A2_minus_A1']
        self.assertEqual(paired['scenes'], 2)
        self.assertEqual(paired['resampling_unit'], 'scene')
        self.assertEqual(paired['metrics']['peak_boxes']['difference'], .5)
        a[0].update(first_reach_step={'3': 100, '4': 200}, steps=300)
        a[1].update(first_reach_step={'3': 150, '4': None}, steps=300)
        late = summarize(a)['third_to_fourth']
        self.assertEqual((late['at_risk'], late['events'], late['censored']), (2, 1, 1))
        self.assertEqual(late['conversion'], .5)
        self.assertEqual(late['km_median_s'], 10.)
        comparisons = paired_comparisons({'A2': b, 'A4-16': a, 'A4-32': a,
                                          'A4-32-T1': a, 'A4-32-T2': b})
        self.assertEqual(comparisons['A4-16_minus_A2']['metrics']['peak_boxes']['difference'], .5)
        self.assertEqual(comparisons['A4-32_minus_A4-16']['metrics']['peak_boxes']['difference'], 0.)
        self.assertEqual(comparisons['A4-32-T1_minus_A4-32']['metrics']['peak_boxes']['difference'], 0.)
        self.assertEqual(comparisons['A4-32-T2_minus_A4-32']['metrics']['peak_boxes']['difference'], -.5)
        self.assertEqual(comparisons['A4-32-T2_minus_A4-32-T1']['metrics']['peak_boxes']['difference'], -.5)

    @unittest.skipIf(torch is None, 'torch is not installed')
    def test_a4_reference_checks_hardware_dtype_and_baseline_identity(self):
        from assignment.training.training.goal_policy import A4_PARENT_SHA256
        from assignment.training.planning.evaluate_goal_actor import load_a2_reference
        keys = ('generate', 'execute', 'n_steps', 'lag', 'switch_tau', 'action_handling', 'goal_switch',
                'tie_break', 'decision_timing', 'A2')
        current = dict(scene_seeds=[100038], sample_seeds=[0], split='val', policy_seed='fixed', task={'horizon': 20},
                       models=dict(world_model=dict(sha256='wm'), goals=dict(sha256='goals')),
                       config={key: 'fixed' for key in keys},
                       runtime=dict(device='cuda', gpus=[0], workers=8, videos=20, versions={},
                                    gpu='NVIDIA L40', dtype='torch.bfloat16'))
        current['config']['candidates'] = {'A2': 16}
        original = json.loads(json.dumps(current))
        original['models']['goal_actor'] = dict(sha256=A4_PARENT_SHA256)
        del original['runtime']['gpu']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'run.json').write_text(json.dumps(original))
            episode = root/'A2/episodes/scene-100038.sample-0'
            episode.mkdir(parents=True)
            row = dict(mode='A2', scene_seed=100038, sample_seed=0,
                       runtime=dict(device='cuda', dtype='torch.bfloat16'))
            (episode/'episode.json').write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'GPU'):
                load_a2_reference(root, current)
            launch = root/'launch.json'
            launch.write_text(json.dumps(dict(checkpoint_sha256=A4_PARENT_SHA256, workers=8, gpu='NVIDIA L40')))
            rows, receipt = load_a2_reference(root, current, launch)
            self.assertEqual(rows, [row])
            self.assertEqual(receipt['gpu'], 'NVIDIA L40')
            original['runtime']['workers'] = 4
            (root/'run.json').write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'workers'):
                load_a2_reference(root, current, launch)

    def test_video_panels_show_goal_and_progress(self):
        from assignment.training.planning.evaluate_goal_actor import video_frame
        current = np.full((128, 256, 3), 10, np.uint8)
        goals = np.stack([np.full_like(current, 20), np.full_like(current, 30)])
        a1 = video_frame(current, goals, 0, 'A1', 100038, 0, 10, 1, 2)
        a3 = video_frame(current, goals, 0, 'A3', 100038, 0, 10, 1, 2)
        self.assertEqual(a1.shape, (352, 256, 3))
        self.assertEqual(a3.shape, (512, 256, 3))
        self.assertEqual(a3[64, 0, 0], 10)
        self.assertEqual(a3[224, 0, 0], 20)
        self.assertEqual(a3[384, 0, 0], 30)


if __name__ == '__main__':
    unittest.main()
