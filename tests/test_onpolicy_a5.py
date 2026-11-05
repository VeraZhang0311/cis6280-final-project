"""A5 invariants: real branch restoration, full-rollout gradients, ranking and frozen state."""
from collections import deque
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch is not installed')
class A5Tests(unittest.TestCase):
    def toy(self):
        class Dynamics(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = type('Config', (), {'context': 3})()
                self.encoder = nn.Linear(2, 2)
                self.projector = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                self.predictor = nn.Linear(2, 2)
                self.action_encoder = nn.Linear(2, 2)
                self.pred_proj = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                self.inverse = nn.Linear(2, 2)
                self.register_buffer('pixel_mean', torch.zeros(1))
                self.register_buffer('pixel_std', torch.ones(1))
            def predict(self, context, action):
                return self.pred_proj((self.predictor(context)+action).flatten(0, 1)).reshape_as(context)
        torch.manual_seed(7)
        return Dynamics()

    def test_full_autoregressive_graph_and_microbatch_weighting(self):
        from assignment.training.training.onpolicy_a5 import freeze_dynamics, rollout
        from assignment.training.training.finetune_goal_onpolicy import backward_batch
        model = self.toy()
        freeze_dynamics(model)
        context, actions, target = torch.randn(5, 3, 2), torch.randn(5, 18, 2), torch.randn(5, 16, 2)
        reference = copy.deepcopy(model)
        # A terminal-only loss must differentiate through all earlier generated states.
        differentiable_actions = actions.clone().requires_grad_()
        prediction = rollout(model, context, differentiable_actions, grad=True)
        prediction[:, -1].sum().backward()
        self.assertGreater(float(differentiable_actions.grad[:, 2].abs().sum()), 0)
        model.zero_grad(set_to_none=True)
        full = (rollout(reference, context, actions, grad=True)-target).square().mean()
        full.backward()
        actual = backward_batch(model, context, actions, target, microbatch=2)
        self.assertAlmostEqual(actual, float(full.detach()), places=5)
        for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6, msg=name)
            else:
                self.assertIsNone(p.grad)

    def test_frozen_parameters_and_bn_buffers_survive_training(self):
        from assignment.training.training.onpolicy_a5 import freeze_dynamics, frozen_digest, representation_digest
        from assignment.training.training.finetune_goal_onpolicy import backward_batch
        model = self.toy()
        parameters = freeze_dynamics(model)
        before, coordinates = frozen_digest(model), representation_digest(model)
        train_before = model.action_encoder.weight.detach().clone()
        self.assertFalse(model.projector.training)
        self.assertFalse(model.pred_proj[1].training)
        optimizer = torch.optim.AdamW(parameters, lr=.01)
        backward_batch(model, torch.randn(4, 3, 2), torch.randn(4, 18, 2), torch.randn(4, 16, 2), 2)
        optimizer.step()
        self.assertEqual(frozen_digest(model), before)
        self.assertEqual(representation_digest(model), coordinates)
        self.assertFalse(torch.equal(model.action_encoder.weight, train_before))
        model.projector[1].running_mean.add_(1)
        self.assertNotEqual(frozen_digest(model), before)

    def test_ranking_uses_trajectory_min_and_nested_candidate_sets(self):
        from assignment.training.training.onpolicy_a5 import ranking_metrics
        target = torch.full((1, 32, 16, 1), 10.)
        target[:, 2, 0] = .1   # Best within the first sixteen; poor terminal distance.
        target[:, 19, 5] = 0.  # Best among all thirty-two.
        predicted = target.clone()
        result = ranking_metrics(predicted, target, torch.zeros(1, 1))
        self.assertEqual(result['agreement_16'], 1.)
        self.assertEqual(result['agreement_32'], 1.)
        self.assertEqual(result['selection_regret'], 0.)
        predicted[:, 0, 3] = 0.
        result = ranking_metrics(predicted, target, torch.zeros(1, 1))
        self.assertGreater(result['regret_16'], 0.)
        self.assertGreater(result['regret_32'], 0.)
        self.assertEqual(result['selection_regret'], (result['regret_16']+result['regret_32'])/2)
        tied = ranking_metrics(torch.ones_like(target), torch.ones_like(target), torch.zeros(1, 1))
        self.assertEqual(tied['selection_regret'], 0.)
        self.assertEqual(tied['agreement_16'], 1.)

    def test_scene_plan_rejects_leakage_and_unequal_budgets(self):
        from assignment.training.training.onpolicy_a5 import SCHEMA, validate_scene_plan
        from assignment.training.data.signals import scene_split
        seeds = [s for s in range(100) if scene_split(s) == 'train']
        plan = dict(schema=SCHEMA, scenes=dict(train=seeds[:4], calibration=seeds[4:6]),
                    excluded=dict(development=[100038], reserved=[100200]),
                    anchors=dict(train=5000, calibration=500), horizon=1795, keep=.25)
        validate_scene_plan(plan)
        overlap = copy.deepcopy(plan)
        overlap['scenes']['calibration'].append(seeds[0])
        with self.assertRaises(ValueError):
            validate_scene_plan(overlap)
        excluded = copy.deepcopy(plan)
        excluded['excluded']['reserved'] = [seeds[0]]
        with self.assertRaises(ValueError):
            validate_scene_plan(excluded)
        with self.assertRaises(ValueError):
            validate_scene_plan(dict(plan, anchors=dict(train=4999, calibration=500)))

    def test_a5_modes_require_selected_wm_and_use_a2_score(self):
        from assignment.training.planning.goal_actor import select_candidate
        distances = torch.full((32, 16), 4.)
        distances[20, 0] = .1
        distances[0, -1] = .2
        self.assertEqual(select_candidate(distances, 'A5-32', 1.).best, 20)
        self.assertEqual(select_candidate(distances[:16], 'A5-16', 1.).best, 0)
        from assignment.training.training.onpolicy_a5 import validate_a5_wm
        with self.assertRaises(ValueError):
            validate_a5_wm(self.toy(), {}, __file__, __file__, None)

    def test_user_checkpoint_requires_explicit_hashed_receipt_without_rewriting_selection(self):
        from assignment.training.training import onpolicy_a5 as module
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wm, policy, selected, requested = [root/name for name in ('wm.pt', 'actor.pt', 'selection.json', 'request.json')]
            wm.write_bytes(b'intermediate WM fixture')
            model = self.toy()
            info = dict(step=56000, a5=dict(schema=module.SCHEMA, actor_sha256=module.ACTOR_SHA256,
                        original_wm_sha256='original-wm', representation_sha256=module.representation_digest(model),
                        frozen_checks=True, selected_step=0, selection=dict(rule='mean_regret_16_32')))
            module.write_json(selected, dict(step=0, sha256='original-selected-checkpoint',
                                             training_complete=True, frozen_checks=True, trained_steps=60000))
            original_bytes = selected.read_bytes()
            digest = module.sha256
            receipt = dict(rule='user_requested_checkpoint', step=56000, sha256=digest(wm),
                           training_selection_sha256=digest(selected), actor_sha256=module.ACTOR_SHA256,
                           original_wm_sha256='original-wm')
            module.write_json(requested, receipt)
            def identity(path):
                return module.ACTOR_SHA256 if Path(path) == policy else digest(path)
            with mock.patch.object(module, 'sha256', side_effect=identity):
                with self.assertRaises(ValueError):
                    module.validate_a5_wm(model, info, policy, wm, selected)
                module.validate_a5_wm(model, info, policy, wm, selected, requested)
                for wrong in (dict(step=54000), dict(sha256='other'), dict(training_selection_sha256='changed')):
                    module.write_json(requested, dict(receipt, **wrong))
                    with self.assertRaisesRegex(ValueError, 'user-requested'):
                        module.validate_a5_wm(model, info, policy, wm, selected, requested)
            self.assertEqual(selected.read_bytes(), original_bytes)
            self.assertEqual(info['a5']['selected_step'], 0)

    def test_a5_paired_reporting(self):
        from assignment.training.planning.evaluate_goal_actor import paired_comparisons
        base = [dict(scene_seed=1, sample_seed=0, peak_boxes=2, final_boxes=1, success=False),
                dict(scene_seed=2, sample_seed=0, peak_boxes=3, final_boxes=2, success=False)]
        improved = [dict(row, final_boxes=row['final_boxes']+1) for row in base]
        result = paired_comparisons({'A4-16': base, 'A4-32': base, 'A5-16': improved, 'A5-32': improved})
        self.assertEqual(result['A5-16_minus_A4-16']['metrics']['final_boxes']['difference'], 1.)
        self.assertEqual(result['A5-32_minus_A4-32']['metrics']['final_boxes']['difference'], 1.)
        self.assertEqual(result['A5-32_minus_A5-16']['metrics']['final_boxes']['difference'], 0.)

    def test_a4_reference_requires_both_complete_matched_budget_groups(self):
        from assignment.training.planning.evaluate_goal_actor import load_a4_reference
        from assignment.training.training.onpolicy_a5 import ACTOR_SHA256, write_json
        keys = ('generate', 'execute', 'n_steps', 'lag', 'switch_tau', 'action_handling', 'goal_switch',
                'tie_break', 'decision_timing', 'A2')
        current = dict(groups=['A5-16', 'A5-32'], scene_seeds=[100038], sample_seeds=[0], split='val',
                       policy_seed='fixed', task={}, actor_config={}, actor_interface={},
                       models=dict(goal_actor=dict(sha256=ACTOR_SHA256), goals=dict(sha256='goals')),
                       config={key: 'fixed' for key in keys},
                       runtime=dict(device='cuda', gpus=[0], workers=8, videos=20, versions={},
                                    gpu='NVIDIA L40', dtype='torch.bfloat16'))
        original = copy.deepcopy(current)
        original.update(groups=['A4-16', 'A4-32'], training_source=dict(cumulative_step=120000))
        original['models']['world_model'] = dict(sha256='original-wm')
        original['config']['candidates'] = {'A4-16': 16, 'A4-32': 32}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root/'run.json', original)
            for mode in original['groups']:
                folder = root/mode/'episodes/scene-100038.sample-0'
                folder.mkdir(parents=True)
                write_json(folder/'episode.json', dict(mode=mode, scene_seed=100038, sample_seed=0,
                                                       runtime=dict(device='cuda', dtype='torch.bfloat16')))
            info = {'a5': {'original_wm_sha256': 'original-wm'}}
            groups, _ = load_a4_reference(root, current, info)
            self.assertEqual(set(groups), {'A4-16', 'A4-32'})
            original['runtime']['workers'] = 4
            write_json(root/'run.json', original)
            with self.assertRaisesRegex(ValueError, 'workers'):
                load_a4_reference(root, current, info)

    def test_completed_dataset_training_and_exact_interruption_recovery(self):
        from assignment.training.training import finetune_goal_onpolicy as trainer
        from assignment.training.training.jepa import JEPAConfig, StackJEPA, save_model, load_model
        from assignment.training.training.onpolicy_a5 import (
            ACTOR_SHA256, SCHEMA, freeze_dynamics, representation_digest, sha256, write_json, load_fixed_dataset,
        )
        from assignment.training.data.signals import scene_split
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'data'
            data.mkdir()
            original = root/'original.pt'
            policy = root/'actor.pt'
            policy.write_bytes(b'fixture actor identity')
            model = StackJEPA(JEPAConfig(dim=6, action_smoothed=4, image=(8, 8), patch=8))
            save_model(original, model, {})
            seeds = [s for s in range(20) if scene_split(s) == 'train']
            plan = dict(schema=SCHEMA, scenes=dict(train=seeds[:2], calibration=seeds[2:4]),
                        excluded=dict(development=[], reserved=[]), anchors=dict(train=4, calibration=2),
                        horizon=1795, keep=.25,
                        models=dict(actor=dict(sha256=ACTOR_SHA256), world_model=dict(sha256=sha256(original))))
            write_json(data/'manifest.json', dict(plan=plan))
            rows = []
            rng = np.random.default_rng(0)
            for split, n, k in (('train', 4, 16), ('calibration', 2, 32)):
                path = data/f'{split}.npz'
                scene = np.resize(np.asarray(plan['scenes'][split]), n)
                np.savez(path, scene=scene, sample_seed=np.zeros(n, np.int32), step=np.arange(n)*5,
                         goal=np.zeros(n, np.int16), behavior=np.repeat([16, 32], n//2),
                         ctx=rng.normal(size=(n, 3, 6)).astype('float32'),
                         past=rng.normal(size=(n, 2, 125)).astype('float32'),
                         chunks=rng.normal(size=(n, k, 16, 125)).astype('float32'),
                         tgt=rng.normal(size=(n, k, 16, 6)).astype('float32'),
                         goal_latent=rng.normal(size=(n, 6)).astype('float32'))
                rows.append(dict(path=path.name, sha256=sha256(path), split=split))
            write_json(data/'complete.json', dict(manifest_sha256=sha256(data/'manifest.json'), shards=rows,
                                                  tasks=[dict(representation_sha256=representation_digest(model))]))
            loaded, _, _ = load_fixed_dataset(data)
            self.assertEqual(loaded['calibration']['chunks'].shape, (2, 32, 16, 125))
            def identity(path):
                return ACTOR_SHA256 if Path(path) == policy else sha256(path)
            base = ['--data', str(data), '--init', str(original), '--policy', str(policy), '--steps', '2',
                    '--batch', '2', '--microbatch', '1', '--warmup', '1', '--eval-every', '1',
                    '--state-every', '1', '--device', 'cpu']
            full, interrupted = root/'full', root/'interrupted'
            with mock.patch.object(trainer, 'sha256', side_effect=identity):
                trainer.main(base+['--out', str(full)])
                actual_save = trainer.atomic_torch_save
                def save_then_interrupt(path, payload):
                    actual_save(path, payload)
                    if payload['step'] == 1:
                        raise RuntimeError('test interruption')
                with mock.patch.object(trainer, 'atomic_torch_save', side_effect=save_then_interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                        trainer.main(base+['--out', str(interrupted)])
                trainer.main(base+['--out', str(interrupted), '--resume'])
            full_state = torch.load(full/'train-state.pt', weights_only=False)
            resumed = torch.load(interrupted/'train-state.pt', weights_only=False)
            self.assertEqual(resumed['step'], 2)
            for key in full_state['model']:
                self.assertTrue(torch.equal(full_state['model'][key], resumed['model'][key]), key)
            self.assertEqual(json.loads((full/'selection.json').read_text())['step'],
                             json.loads((interrupted/'selection.json').read_text())['step'])
            selected, info = load_model(interrupted/'model.pt')
            self.assertEqual(info['a5']['selected_step'], info['step'])
            self.assertEqual(representation_digest(selected), representation_digest(model))
            from assignment.training.training import onpolicy_a5
            with mock.patch.object(onpolicy_a5, 'sha256', side_effect=identity):
                onpolicy_a5.validate_a5_wm(selected, info, policy, interrupted/'model.pt', interrupted/'selection.json')
                selected.pixel_mean.add_(1)
                with self.assertRaises(ValueError):
                    onpolicy_a5.validate_a5_wm(selected, info, policy, interrupted/'model.pt', interrupted/'selection.json')
            # Any post-collection mutation is rejected before optimizer construction.
            with (data/'train.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Shard identity'):
                load_fixed_dataset(data)


@unittest.skipIf(torch is None, 'torch is not installed')
class SnapshotTests(unittest.TestCase):
    def test_real_mujoco_state_history_controls_and_failed_branches_restore(self):
        import mujoco
        from assignment.training.data.collect_goal_onpolicy import snapshot, restore, execute_branches
        class Sim:
            def __init__(self):
                self.model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
                    <worldbody><body><joint type="slide"/><geom size=".1" mass="1"/></body></worldbody>
                    <actuator><motor joint="0"/></actuator></mujoco>'''.replace('joint="0"', 'joint="slide"')
                    .replace('<joint type=', '<joint name="slide" type='))
                self.data = mujoco.MjData(self.model)
                self.targets = np.array([.3])
                self.elapsed_steps = 25
                self.start_time = 0.
                self.status, self.stop_reason, self.finalization_steps = 'running', None, 0
                self.history = deque([np.array([i]) for i in range(31)], maxlen=31)
            def step(self, action):
                self.data.ctrl[:] = action[0]
                mujoco.mj_step(self.model, self.data)
                self.targets += action[0]
                self.elapsed_steps += 1
                self.history.append(self.data.qpos.copy())
            def render(self):
                return np.full((2, 2, 2, 3), self.elapsed_steps, np.uint8)
        sim = Sim()
        initial = snapshot(sim)
        raw = np.ones((2, 16, 5, 25), np.float32)*.1
        targets = execute_branches(sim, raw, lambda frames: frames[:, 0, 0, :1].astype(np.float32))
        self.assertEqual(targets.shape, (2, 16, 1))
        np.testing.assert_array_equal(targets[0, :, 0], np.arange(30, 106, 5))
        np.testing.assert_array_equal(targets[0], targets[1])
        after = snapshot(sim)
        np.testing.assert_array_equal(after['state'], initial['state'])
        np.testing.assert_array_equal(sim.targets, initial['attributes']['targets'])
        self.assertEqual(sim.elapsed_steps, 25)
        for actual, expected in zip(sim.history, initial['history']):
            np.testing.assert_array_equal(actual, expected)
        original_step = sim.step
        def stop(action):
            original_step(action)
            sim.status = 'finalizing'
        sim.step = stop
        with self.assertRaisesRegex(ValueError, 'incomplete_branch'):
            execute_branches(sim, raw, lambda frames: frames)
        self.assertEqual(sim.status, 'running')
        np.testing.assert_array_equal(snapshot(sim)['state'], initial['state'])


if __name__ == '__main__':
    unittest.main()
