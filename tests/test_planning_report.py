"""Metric definitions, time alignment and representative selection in the offline report."""
from pathlib import Path
import unittest

import numpy as np

from assignment.training.planning.report import choose_representatives, derive_steps, sampled, verify_episodes


class ReportMetrics(unittest.TestCase):
    def test_cross_artifact_verification_rejects_wrong_candidate(self):
        row = dict(control_step=0, sim_time=0., prediction_mse=None, plan_change_rms=None,
                   active_goal_index=0, candidate_scores=list(range(16)), selected_candidate=0,
                   stacked_count=0, max_stacked_count=0)
        episode = dict(folder=Path('fixture'), rows=[row],
                       summary=dict(steps=5, goals=2, max_placed=1, final_count=1),
                       events=[dict(event='episode_end', control_step=5, final_count=1)])
        self.assertEqual(verify_episodes([episode], {})['candidate_scores'], 16)
        row['selected_candidate'] = 4
        with self.assertRaisesRegex(ValueError, 'selected candidate'):
            verify_episodes([episode], {})

    def test_candidate_gap_and_zero_threshold_are_not_confused(self):
        rows = [dict(sim_time=0, candidate_scores=[12., 10., 14.], goal_distance_sq=4., goal_threshold_sq=2.,
                     active_goal_index=1),
                dict(sim_time=.5, candidate_scores=[3., 3., 3.], goal_distance_sq=0., goal_threshold_sq=0.,
                     active_goal_index=2)]
        data = derive_steps(rows, 3)
        np.testing.assert_allclose(data['candidate_best'], [10., 3.])
        np.testing.assert_allclose(data['relative_gap'], [.2, 0.])
        self.assertEqual(data['distance_ratio'][0], 2.)
        self.assertTrue(np.isnan(data['distance_ratio'][1]))
        np.testing.assert_allclose(data['goal_fraction'], [.5, 1.])

    def test_alignment_keeps_missing_observations_missing(self):
        e = dict(data=dict(sim_time=np.array([0., .5, 1.]), stacked_count=np.array([0., 1., 2.])),
                 metric=dict(first_box_s=.2))
        actual = sampled(e, 'stacked_count', np.array([-1., 0., .3, .8, 1.3]), True)
        np.testing.assert_allclose(actual, [np.nan, 0., 1., 2., np.nan], equal_nan=True)

    def test_representatives_are_unique_without_comparing_numpy_arrays(self):
        episodes = [dict(folder=Path(str(i)), data=np.arange(3),
                         metric=dict(peak=i%5, final=i%3, longest_nonfinal_goal_s=10*i,
                                     prediction_mse_p95=i*.01, plan_change_rms_median=i*.02,
                                     candidate_relative_gap_median=i*.1)) for i in range(8)]
        chosen = choose_representatives(episodes)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(len({e['folder'] for _, e in chosen}), 6)


if __name__ == '__main__':
    unittest.main()
