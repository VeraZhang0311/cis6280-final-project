"""Experimental partitioning and paired statistics, using synthetic report fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from assignment.planning.experiments import validation_seeds
from assignment.planning.report import compare, read_reports


class ExperimentTests(unittest.TestCase):
    def test_development_and_confirmation_scenes_are_disjoint(self):
        scenes = validation_seeds()
        self.assertEqual(len(set(scenes)), 300)
        self.assertFalse(set(scenes[:30]) & set(scenes[30:]))
        for seed in scenes:
            bucket = int(hashlib.sha256(f'stacking-split-v1/{seed}'.encode()).hexdigest(), 16) % 100
            self.assertTrue(80 <= bucket < 90)

    def reports(self, root, name, increment=0, seeds=(0, 2, 3)):
        paths = []
        for seed in seeds:
            path = root/f'{name}-{seed}.json'
            path.write_text(json.dumps(dict(
                kind='goal', config={'name': name}, policy='fixed', world_model='fixed',
                split='val', task={'horizon': 1795}, goals='fixed', sample_seed=seed,
                episodes=[dict(scene_seed=scene, max_placed=1+increment, steps=100,
                               ground_falls=0, agent_s=2.) for scene in (100035, 100038)])))
            paths.append(path)
        return paths

    def test_paired_constant_improvement_has_exact_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.reports(root, 'baseline')
            candidate = self.reports(root, 'candidate', increment=1)
            result = compare(baseline, candidate, repeats=100)
            self.assertEqual(result['scenes'], 2)
            self.assertEqual(result['episodes'], 6)
            self.assertEqual(result['paired_delta'], 1)
            self.assertEqual(result['scene_bootstrap_95ci'], [1, 1])

    def test_rejects_duplicate_and_unpaired_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.reports(root, 'baseline')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_reports(baseline+baseline)
            candidate = self.reports(root, 'candidate', seeds=(0,))
            with self.assertRaisesRegex(ValueError, 'identical'):
                compare(baseline, candidate, repeats=100)

    def test_summary_requires_complete_stage_and_selects_best(self):
        from assignment.planning.summarize import summarize_experiment
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, increment in [('baseline', 0), ('combined', 1)]:
                directory = root/'smoke'/name
                directory.mkdir(parents=True)
                self.reports(directory, name, increment=increment, seeds=(0,))
            result = summarize_experiment(root, 'smoke', ['baseline', 'combined'])
            self.assertEqual(result['best_equal_budget'], 'combined')
            self.assertEqual(result['episodes_per_variant'], 2)
            path = root/'smoke/combined/combined-0.json'
            incomplete = json.loads(path.read_text())
            incomplete['episodes'] = incomplete['episodes'][:1]
            path.write_text(json.dumps(incomplete))
            with self.assertRaisesRegex(ValueError, 'coverage'):
                summarize_experiment(root, 'smoke', ['baseline', 'combined'])


if __name__ == '__main__':
    unittest.main()
