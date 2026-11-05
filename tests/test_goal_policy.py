"""Training-only v2 contracts: initialization, future-goal alignment and saved artifacts."""
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
class GoalPolicyTests(unittest.TestCase):
    def setUp(self):
        from assignment.training.training.policy import DFPolicy, PolicyConfig, save_policy
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache, self.latents = self.root/'cache', self.root/'latents'
        self.cache.mkdir()
        self.latents.mkdir()
        episodes, start = [], 0
        specs = [('train', 20, 90, .04), ('train', 18, 85, None), ('val', 19, 90, .02),
                 ('test', 20, 95, 0.), ('train', 20, 95, .05), ('train', 16, 75, .01)]
        for scene, (split, length, control, sigma) in enumerate(specs):
            episodes.append(dict(episode_id=f'ep{scene}', scene_seed=scene, split=split, start=start,
                                 length=length, control_steps=control, sigma=sigma))
            start += length
        self.index = dict(frames=start, stride=5, action_dim=125, action_mean=[.1]*25,
                          action_std=[2.]*25, episodes=episodes)
        self.write_index()
        self.raw_z = (np.arange(start*4).reshape(start, 4)/8).astype(np.float16)
        self.raw_actions = np.repeat(np.arange(start, dtype=np.float32)[:, None]/100, 125, axis=1)
        self.raw_z.tofile(self.latents/'latents.f16')
        self.raw_actions.tofile(self.cache/'actions.f32')
        self.stats = dict(frames=start, dim=4, model='frozen-encoder.pt', mean=[1.]*4, std=[2.]*4)
        (self.latents/'latents.json').write_text(json.dumps(self.stats))
        self.encoder = self.root/'encoder.pt'
        self.encoder.write_bytes(b'not loaded: identity only')
        torch.manual_seed(9)
        self.original = DFPolicy(PolicyConfig(latent_dim=4, dim=16, depth=2, heads=2, mlp=32))
        # Exercise a nontrivial pretrained backbone, not the default all-zero output head.
        with torch.no_grad():
            for parameter in self.original.parameters():
                parameter.normal_(0, .15)
        self.official = self.root/'official.pt'
        self.info = dict(z_mean=[1.]*4, z_std=[2.]*4, action_mean=[.1]*125,
                         action_std=[2.]*125, encoder='frozen-encoder.pt')
        save_policy(self.official, self.original, self.info)

    def write_index(self):
        (self.cache/'index.json').write_text(json.dumps(self.index))

    def data(self, split='train'):
        from assignment.training.training.goal_policy import FutureGoalData
        return FutureGoalData(self.cache, self.latents, split)

    def test_future_goals_and_full_labels_respect_boundaries_and_splits(self):
        train, val = self.data(), self.data('val')
        np.testing.assert_array_equal(train.anchors, [0, 1, 2, 20, 21])
        np.testing.assert_array_equal(val.anchors, [38, 39, 40])
        # K endpoints: nearby goal must still supervise all 16 actions.
        current, goal, actions, mask = train.batch_at([2, 21], [1, 16])
        np.testing.assert_array_equal(current.numpy()[:, 0], (self.raw_z[[2, 21]].astype(np.float32)-1)/2)
        np.testing.assert_array_equal(goal.numpy()[:, 0], (self.raw_z[[3, 37]].astype(np.float32)-1)/2)
        expected = (self.raw_actions[np.array([2, 21])[:, None]+np.arange(16)]-.1)/2
        np.testing.assert_array_equal(actions.numpy(), expected)
        self.assertTrue(mask.all())
        self.assertEqual(tuple(mask.shape), (2, 16))
        for anchors, offsets in (([3], [1]), ([38], [1]), ([0], [0]), ([0], [17])):
            with self.assertRaises(ValueError):
                train.batch_at(anchors, offsets)
        indices, k = train.draw_indices(torch.Generator().manual_seed(3), 32000)
        self.assertTrue(np.isin(indices, train.anchors).all())
        counts = np.bincount(k, minlength=17)[1:]
        self.assertTrue(np.all((counts > 1750) & (counts < 2250)), counts)
        one = train.draw_indices(torch.Generator().manual_seed(8), 80)
        two = train.draw_indices(torch.Generator().manual_seed(8), 80)
        for a, b in zip(one, two):
            np.testing.assert_array_equal(a, b)

    def test_reject_split_leakage_missing_stop_metadata_and_empty_split(self):
        self.index['episodes'][2]['scene_seed'] = 0
        self.write_index()
        with self.assertRaisesRegex(ValueError, 'leaks'):
            self.data()
        self.index['episodes'][2]['scene_seed'] = 2
        del self.index['episodes'][0]['control_steps']
        self.write_index()
        with self.assertRaisesRegex(ValueError, 'control_steps'):
            self.data()
        self.index['episodes'][0]['control_steps'] = 0
        self.index['episodes'][1]['control_steps'] = 0
        self.write_index()
        with self.assertRaisesRegex(ValueError, 'No full-16'):
            self.data()

    def test_zero_adapter_forward_and_sampling_are_exactly_official(self):
        from assignment.training.training.goal_policy import GoalDFPolicy
        from assignment.training.training.jepa import modulate
        from assignment.training.training.policy import timestep_embedding
        model, _ = GoalDFPolicy.from_official(self.official)
        current = torch.randn(2, 1, 4)
        x, t = torch.randn(2, 16, 125), torch.rand(2, 16)
        # Independent original forward reference protects the small backbone refactor.
        old = self.original
        tokens = torch.cat([old.obs_in(current), old.act_in(x)], 1)+old.pos
        cond = torch.cat([old.obs_cond.expand(2, 1, -1), old.t_embed(timestep_embedding(t, 256))], 1)
        for block in old.blocks:
            tokens = block(tokens, cond, old.mask)
        shift, scale = old.out_ada(cond[:, 1:]).chunk(2, -1)
        expected = old.out(modulate(old.norm(tokens[:, 1:]), shift, scale))
        self.assertTrue(torch.equal(old(current, x, t), expected))
        for goal in (torch.randn_like(current), torch.randn_like(current)*10):
            self.assertTrue(torch.equal(model(current, goal, x, t), expected))
            for lag in (0, 1):
                generated = model.sample(current, goal, n_steps=3, lag=lag,
                                         generator=torch.Generator().manual_seed(14))
                reference = old.sample(current, n_steps=3, lag=lag,
                                       generator=torch.Generator().manual_seed(14))
                self.assertTrue(torch.equal(generated, reference))

    def test_tail_anchors_masks_padding_and_uniform_conditional_goals(self):
        from assignment.training.training.goal_policy import FutureGoalData
        data = FutureGoalData(self.cache, self.latents, 'train', include_tail=True)
        self.assertEqual((len(data.anchors), data.full_anchors, data.tail_anchors), (50, 5, 45))
        np.testing.assert_array_equal(data.r_counts, [3]*15+[5])
        self.assertTrue(np.isin([17, 36, 111], data.anchors).all())
        with (self.cache/'actions.f32').open('r+b') as stream:
            stream.seek(18*125*4)
            stream.write(np.full((2, 125), np.nan, np.float32).tobytes())
        current, goal, actions, mask = data.batch_at([17, 15, 111], [1, 1, 1])
        self.assertEqual(mask.sum(-1).tolist(), [1, 3, 1])
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue((actions[~mask] == 0).all())
        np.testing.assert_array_equal(goal.numpy()[:, 0], (self.raw_z[[18, 16, 112]].astype(np.float32)-1)/2)
        np.testing.assert_array_equal(actions[1, :3].numpy(), (self.raw_actions[15:18]-.1)/2)
        for anchors, offsets in (([17], [2]), ([18], [1]), ([38], [1]), ([77], [1])):
            with self.assertRaises(ValueError):
                data.batch_at(anchors, offsets)
        anchors, k = data.draw_indices(torch.Generator().manual_seed(9), 80000)
        lengths = data.valid_lengths(anchors)
        self.assertTrue((k <= lengths).all())
        for r in (1, 3, 8, 16):
            selected = k[lengths == r]
            counts = np.bincount(selected, minlength=r+1)[1:r+1]
            self.assertTrue(np.all(np.abs(counts-len(selected)/r) < .25*len(selected)/r))
        val = FutureGoalData(self.cache, self.latents, 'val', include_tail=True)
        self.assertEqual((len(val.anchors), val.full_anchors, val.tail_anchors), (18, 3, 15))

    def test_masked_accumulation_matches_full_valid_block_objective(self):
        from assignment.training.training.goal_policy import FutureGoalData, GoalDFPolicy
        from assignment.training.training.train_goal_policy import backward_batch, evaluate
        data = FutureGoalData(self.cache, self.latents, 'train', include_tail=True)
        anchors, offsets = np.array([0, 15, 17]), np.array([1, 1, 1])
        current, goal, actions, mask = data.batch_at(anchors, offsets)
        class ScalarPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.2))
            def loss(self, current, goal, actions, mask, generator=None):
                return ((self.weight-actions).square().mean(-1)*mask).sum()/mask.sum()
            def sample(self, current, goal, **kwargs):
                return torch.ones(len(current), 16, 125)
        expected_loss = float((.2-actions[mask]).square().mean())
        expected_grad = 2*(.2-float(actions[mask].mean()))
        for micro in (1, 2, 3):
            scalar = ScalarPolicy()
            loss = backward_batch(scalar, data, anchors, offsets, micro, torch.Generator())
            self.assertAlmostEqual(loss, expected_loss, places=6)
            self.assertAlmostEqual(float(scalar.weight.grad), expected_grad, places=6)
        model, _ = GoalDFPolicy.from_official(self.official)
        altered = actions.clone()
        altered[~mask] = 100
        original_loss = model.loss(current, goal, actions, mask, torch.Generator().manual_seed(41))
        padded_loss = model.loss(current, goal, altered, mask, torch.Generator().manual_seed(41))
        torch.testing.assert_close(original_loss, padded_loss)
        drawn, k = data.draw_indices(torch.Generator().manual_seed(1234), 8)
        _, _, labels, valid = data.batch_at(drawn, k)
        metrics = evaluate(ScalarPolicy(), data, 8, 3)
        self.assertAlmostEqual(metrics['val_loss'], float(((.2-labels).square().mean(-1)*valid).sum()/valid.sum()), places=6)
        self.assertAlmostEqual(metrics['mse_h16_lag0_all'], float((1-labels[valid]).square().mean()), places=6)
        self.assertEqual(metrics['valid_blocks'], int(valid.sum()))

    def test_tail_phase_uses_fresh_rng_and_recovers_sampling_statistics(self):
        from assignment.training.training import train_goal_policy as trainer
        from assignment.training.training.goal_policy import load_goal_policy
        base = ['--cache', str(self.cache), '--latents', str(self.latents), '--encoder', str(self.encoder),
                '--device', 'cpu', '--steps', '2', '--warmup', '1', '--batch', '3', '--microbatch', '2',
                '--val-batch', '3', '--state-every', '1', '--lr', '1e-4']
        parent = self.root/'parent'
        trainer.main([*base, '--init-policy', str(self.official), '--out', str(parent)])
        parent_model, _ = load_goal_policy(parent/'policy.pt')
        parent_hash = trainer.sha256(parent/'policy.pt')
        args = [*base, '--include-tail', '--init-goal-policy', str(parent/'policy.pt'),
                '--expected-init-sha256', parent_hash, '--expected-prior-steps', '2']
        real_save = trainer.save_training_state
        initial = []
        def inspect_initial(*items, **kwargs):
            if items[7] == 0:
                for key, value in parent_model.state_dict().items():
                    self.assertTrue(torch.equal(value, items[1].state_dict()[key]), key)
                self.assertEqual(len(items[3].state), 0)
                self.assertTrue(torch.equal(items[5].get_state(), torch.Generator().manual_seed(0).get_state()))
                self.assertTrue(torch.equal(items[6].get_state(), torch.Generator().manual_seed(0).get_state()))
                initial.append(True)
            real_save(*items, **kwargs)
        with mock.patch.object(trainer, 'save_training_state', side_effect=inspect_initial):
            trainer.main([*args, '--out', str(self.root/'tail-full')])
        self.assertEqual(initial, [True])
        class Interrupted(Exception):
            pass
        def interrupt(*items, **kwargs):
            real_save(*items, **kwargs)
            if items[7] == 1:
                raise Interrupted()
        recovery = [*args, '--out', str(self.root/'tail-recovered')]
        with mock.patch.object(trainer, 'save_training_state', side_effect=interrupt):
            with self.assertRaises(Interrupted):
                trainer.main(recovery)
        trainer.main([*recovery, '--resume'])
        full, full_info = load_goal_policy(self.root/'tail-full/policy.pt')
        recovered, info = load_goal_policy(self.root/'tail-recovered/policy.pt')
        for key, value in full.state_dict().items():
            self.assertTrue(torch.equal(value, recovered.state_dict()[key]), key)
        self.assertEqual(info['sampling_observed'], full_info['sampling_observed'])
        self.assertEqual(info['sampling_observed']['samples'], 6)
        self.assertEqual(sum(info['sampling_observed']['R_counts']), 6)
        joint = np.asarray(info['sampling_observed']['RK_counts'])
        self.assertEqual(int(joint.sum()), 6)
        self.assertTrue((joint[np.triu_indices(16, 1)] == 0).all())
        self.assertEqual(info['data']['train_tail_anchors'], 45)
        self.assertEqual(info['rng']['initialization'], 'manual_seed')
        self.assertEqual(parent_hash, trainer.sha256(parent/'policy.pt'))
        trainer.main([*recovery, '--resume'])  # A restart after the final state must retain validation evidence.
        _, completed = load_goal_policy(self.root/'tail-recovered/policy.pt')
        self.assertEqual(completed['validation_sampling'], full_info['validation_sampling'])
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            trainer.main([*args, '--out', str(self.root/'bad-parent'), '--expected-init-sha256', 'wrong'])

    def test_goal_features_gradients_and_original_full_chunk_objective(self):
        from assignment.training.training.goal_policy import GoalDFPolicy
        from assignment.training.training.train_goal_policy import optimizer_for
        model, _ = GoalDFPolicy.from_official(self.official)
        current, goal, actions, mask = self.data().batch_at([1, 20], [1, 16])
        captured = []
        hook = model.goal_adapter.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
        generator = torch.Generator().manual_seed(31)
        loss = model.loss(current, goal, actions, mask, generator)
        hook.remove()
        delta = goal-current
        self.assertTrue(torch.equal(captured[0], torch.cat((delta, current*delta), -1)))
        generator.manual_seed(31)
        t = torch.rand(2, 16, generator=generator)
        eps = torch.randn(actions.shape, generator=generator)
        reference = (model(current, goal, (1-t[..., None])*actions+t[..., None]*eps, t)-(eps-actions)).square().mean()
        torch.testing.assert_close(loss, reference)
        # Altering only actions after the nearby K=1 goal must change the supervised objective.
        altered = actions.clone()
        altered[:, 1:] += 2
        changed = model.loss(current, goal, altered, mask, torch.Generator().manual_seed(31))
        self.assertNotEqual(float(loss.detach()), float(changed.detach()))
        opt = optimizer_for(model, 1e-3, .01, torch.device('cpu'))
        optimized = {id(p) for group in opt.param_groups for p in group['params']}
        self.assertEqual(optimized, {id(p) for p in model.parameters()})
        loss.backward()
        self.assertGreater(float(model.goal_adapter[-1].weight.grad.abs().sum()), 0)
        before = model.obs_in[0].weight.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(before, model.obs_in[0].weight))
        self.assertFalse(any(name.startswith(('encoder.', 'projector.', 'predictor.'))
                             for name, _ in model.named_parameters()))

    def test_checkpoint_is_typed_roundtrips_and_rejects_wrong_loaders(self):
        from assignment.training.training.goal_policy import GoalDFPolicy, load_goal_policy, save_goal_policy
        from assignment.training.training.policy import load_policy
        model, _ = GoalDFPolicy.from_official(self.official, adapter_hidden=12)
        path = self.root/'goal.pt'
        save_goal_policy(path, model, {'chosen': 'final_step_ema'})
        loaded, info = load_goal_policy(path)
        self.assertEqual(info, {'chosen': 'final_step_ema'})
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
        with self.assertRaisesRegex(ValueError, 'load_goal_policy'):
            load_policy(path)
        with self.assertRaisesRegex(ValueError, 'explicitly typed'):
            load_goal_policy(self.official)
        ckpt = torch.load(path, weights_only=False)
        ckpt['interface']['goal_features'] = 'different_features'
        torch.save(ckpt, path)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            load_goal_policy(path)

    def test_two_step_cpu_training_records_recipe_and_preserves_official(self):
        from assignment.training.training.goal_policy import load_goal_policy
        from assignment.training.training.train_goal_policy import main, sha256
        original_hash = sha256(self.official)
        out, logs = self.root/'trained', self.root/'logs'
        arguments = ['--cache', str(self.cache), '--latents', str(self.latents),
                     '--init-policy', str(self.official), '--encoder', str(self.encoder),
                     '--out', str(out), '--log-dir', str(logs), '--device', 'cpu',
                     '--steps', '2', '--warmup', '1', '--batch', '3', '--microbatch', '2',
                     '--val-batch', '3', '--eval-every', '1', '--save-every', '1']
        with mock.patch('assignment.training.training.jepa.load_model', side_effect=AssertionError('WM loaded')):
            main(arguments)
        model, info = load_goal_policy(out/'policy.pt')
        self.assertEqual(info['step'], 2)
        self.assertEqual(sha256(self.official), original_hash)
        self.assertEqual(info['checkpoint_selection']['rule'], 'final_step_ema')
        self.assertEqual(info['optimization']['effective_batch'], 3)
        self.assertEqual(info['optimization']['accumulation_steps'], 2)
        self.assertEqual(info['data']['train_anchors'], 5)
        self.assertEqual(info['data']['val_anchors'], 3)
        self.assertIn('source_sha256', info['code'])
        self.assertEqual(json.loads((out/'run.json').read_text()), json.loads((logs/'run.json').read_text()))
        rows = [json.loads(line) for line in (logs/'train-log.jsonl').read_text().splitlines()]
        self.assertEqual([row['step'] for row in rows], [1, 2])
        self.assertTrue(all(np.isfinite(row['eval']['val_loss']) for row in rows))
        self.assertGreater(float(model.goal_adapter[-1].weight.detach().abs().sum()), 0)
        step, _ = load_goal_policy(out/'step-000002.pt')
        for key, value in step.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            main(arguments)

    def test_interrupted_training_recovers_optimizer_ema_and_rng(self):
        from assignment.training.training import train_goal_policy as trainer
        from assignment.training.training.goal_policy import load_goal_policy
        base = ['--cache', str(self.cache), '--latents', str(self.latents),
                '--init-policy', str(self.official), '--encoder', str(self.encoder),
                '--device', 'cpu', '--steps', '2', '--warmup', '1', '--batch', '3',
                '--microbatch', '2', '--val-batch', '2', '--state-every', '1']
        trainer.main([*base, '--out', str(self.root/'continuous')])
        real_save = trainer.save_training_state
        class Interrupted(Exception):
            pass
        def interrupt_after_step_one(*args, **kwargs):
            real_save(*args, **kwargs)
            if args[7] == 1:
                raise Interrupted()
        resumed_args = [*base, '--out', str(self.root/'recovered')]
        with mock.patch.object(trainer, 'save_training_state', side_effect=interrupt_after_step_one):
            with self.assertRaises(Interrupted):
                trainer.main(resumed_args)
        trainer.main([*resumed_args, '--resume'])
        continuous, _ = load_goal_policy(self.root/'continuous/policy.pt')
        recovered, info = load_goal_policy(self.root/'recovered/policy.pt')
        self.assertEqual(info['resumed_from_step'], 1)
        for key, value in continuous.state_dict().items():
            self.assertTrue(torch.equal(value, recovered.state_dict()[key]), key)
        with self.assertRaisesRegex(ValueError, 'recipe differs'):
            trainer.main([*resumed_args, '--resume', '--lr', '1e-4'])

    def test_new_phase_keeps_trained_adapter_resets_optimizer_and_continues_rng(self):
        from assignment.training.training import train_goal_policy as trainer
        from assignment.training.training.goal_policy import load_goal_policy
        base = ['--cache', str(self.cache), '--latents', str(self.latents),
                '--encoder', str(self.encoder), '--device', 'cpu', '--steps', '2', '--warmup', '1',
                '--batch', '3', '--microbatch', '2', '--val-batch', '2', '--state-every', '1']
        first, second = self.root/'phase1', self.root/'phase2'
        trainer.main([*base, '--init-policy', str(self.official), '--out', str(first)])
        parent_model, _ = load_goal_policy(first/'policy.pt')
        parent_state = torch.load(first/'train-state.pt', weights_only=False)
        parent_hash = trainer.sha256(first/'policy.pt')
        real_save = trainer.save_training_state
        initializations = []
        def check_initial_state(*args, **kwargs):
            if args[7] == 0:
                _, model, ema, optimizer, scheduler, data_gen, flow_gen, _, info = args
                self.assertEqual(len(optimizer.state), 0)
                self.assertEqual(info['optimization']['lr'], 1e-4)
                self.assertEqual(info['prior_optimizer_steps'], 2)
                for key, value in parent_model.state_dict().items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
                    self.assertTrue(torch.equal(value, ema.state_dict()[key]), key)
                self.assertTrue(torch.equal(data_gen.get_state(), parent_state['data_rng']))
                self.assertTrue(torch.equal(flow_gen.get_state(), parent_state['flow_rng']))
                initializations.append(True)
            real_save(*args, **kwargs)
        with mock.patch.object(trainer, 'save_training_state', side_effect=check_initial_state):
            trainer.main([*base, '--init-goal-policy', str(first/'policy.pt'),
                          '--init-training-state', str(first/'train-state.pt'), '--out', str(second), '--lr', '1e-4'])
        self.assertEqual(initializations, [True])
        _, info = load_goal_policy(second/'policy.pt')
        self.assertEqual(info['training_phase'], 2)
        self.assertEqual(info['step'], 2)
        self.assertEqual(info['cumulative_step'], 4)
        self.assertEqual(info['rng']['initialization'], 'parent_phase_end')
        self.assertEqual(trainer.sha256(first/'policy.pt'), parent_hash)

    def test_wandb_tracking_records_identity_and_failure_status(self):
        from assignment.training.training.train_goal_policy import parser, tracking
        args = parser().parse_args(['--cache', 'cache', '--latents', 'latents', '--init-policy', 'actor.pt',
                                    '--encoder', 'encoder.pt', '--out', str(self.root/'out'),
                                    '--wandb-project', '6280_actor', '--wandb-run-id', 'fixed-id'])
        tracker = mock.Mock(project='6280_actor', entity='my-entity', id='fixed-id',
                            url='https://wandb.ai/my-entity/6280_actor/runs/fixed-id')
        tracker.summary = {}
        sdk = types.SimpleNamespace(init=mock.Mock(return_value=tracker), __version__='test-sdk')
        info = {key: {} for key in ('model_type', 'config', 'interface', 'optimization', 'rng',
                                     'runtime', 'init_policy', 'checkpoint_selection')}
        info.update(data=dict(train_episodes=['do not upload'], val_episodes=['do not upload'],
                              train_anchors=5), code=dict(commit='abc', branch='test', dirty=True))
        with mock.patch.dict('sys.modules', wandb=sdk):
            with tracking(args, self.root, info) as active:
                self.assertIs(active, tracker)
            self.assertEqual(info['wandb']['id'], 'fixed-id')
            self.assertNotIn('train_episodes', sdk.init.call_args.kwargs['config']['data'])
            tracker.finish.assert_called_with(exit_code=0)
            with self.assertRaisesRegex(RuntimeError, 'failed training'):
                with tracking(args, self.root, info):
                    raise RuntimeError('failed training')
            tracker.finish.assert_called_with(exit_code=1)


if __name__ == '__main__':
    unittest.main()
