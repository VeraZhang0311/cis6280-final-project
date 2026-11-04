"""The capture wrapper must observe existing inference without extra model calls."""
from types import SimpleNamespace
import unittest

import numpy as np

from assignment.training.planning.openloop import CaptureInference


class Model:
    def __init__(self):
        self.sample_calls = self.rollout_calls = 0
    def sample(self):
        self.sample_calls += 1
        return np.ones((16, 16, 125), np.float32)
    def rollout(self):
        self.rollout_calls += 1
        return np.ones((16, 16, 192), np.float32)


class CaptureTests(unittest.TestCase):
    def test_prefix_and_measurement_are_separate_and_wrapping_preserves_output(self):
        model = Model()
        agent = SimpleNamespace(policy=model, encoder=model)
        original = model.sample
        with CaptureInference(agent) as capture:
            model.sample()
            model.rollout()
            capture.phase = 'test'
            blocks, predicted = model.sample(), model.rollout()
            np.testing.assert_array_equal(capture.blocks, blocks)
            np.testing.assert_array_equal(capture.imagined, predicted)
            self.assertEqual(model.sample_calls, 2)
            self.assertEqual(model.rollout_calls, 2)
            self.assertEqual(capture.calls, dict(prefix_sample=1, prefix_rollout=1, test_sample=1, test_rollout=1))
        self.assertEqual(model.sample, original)

    def test_second_rollout_is_rejected_before_the_model_runs(self):
        model = Model()
        with CaptureInference(SimpleNamespace(policy=model, encoder=model)) as capture:
            capture.phase = 'test'
            model.rollout()
            with self.assertRaises(RuntimeError):
                model.rollout()
            self.assertEqual(model.rollout_calls, 1)
            capture.phase = 'observe'
            with self.assertRaises(RuntimeError):
                model.sample()
            self.assertEqual(model.sample_calls, 0)


if __name__ == '__main__':
    unittest.main()
