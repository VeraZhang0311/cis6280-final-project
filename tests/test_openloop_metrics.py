"""Future-block alignment and per-latent goal distance definitions."""
import unittest

import numpy as np

from assignment.training.planning.openloop_metrics import (
    aligned_trajectory_metrics, candidate_goal_distances,
)


class OpenLoopMetrics(unittest.TestCase):
    def test_per_step_scores_preserve_terminal_selection(self):
        imagined = np.array([[[3.], [0.]], [[1.], [1.]]])
        distances = candidate_goal_distances(imagined, np.array([0.]))
        self.assertEqual(int(distances[:, -1].argmin()), 0)
        np.testing.assert_allclose(distances, [[9., 0.], [1., 1.]])

    def test_mse_and_goal_distance_use_different_reductions(self):
        imagined = np.array([[1., 2.], [3., 4.]])
        actual = np.array([[0., 0.], [1., 2.]])
        metrics = aligned_trajectory_metrics(imagined, actual, np.zeros(2))
        np.testing.assert_allclose(metrics['latent_mse'], [2.5, 4.])
        np.testing.assert_allclose(metrics['imagined_goal_distance_sq'], [5., 25.])
        np.testing.assert_allclose(metrics['actual_goal_distance_sq'], [0., 5.])

    def test_context_frame_cannot_silently_shift_the_actual_trajectory(self):
        with self.assertRaises(ValueError):
            aligned_trajectory_metrics(np.zeros((16, 192)), np.zeros((17, 192)), np.zeros(192))


if __name__ == '__main__':
    unittest.main()
