import unittest
from assignment.training.planning.sampling import stop_reason, improvement


class StopRules(unittest.TestCase):
    def test_both_conditions_required_to_resample_and_strict_threshold(self):
        self.assertIsNone(stop_reason(100.,81.,81.,16))
        self.assertEqual(stop_reason(100.,81.,82.,16),'predicted_threshold')
        self.assertEqual(stop_reason(100.,80.,1.,16),'sufficient_improvement')
        self.assertIsNone(stop_reason(100.,81.,0.,32))

    def test_zero_current_distance_and_zero_threshold(self):
        self.assertEqual(stop_reason(1e-8,1.,0.,16),'near_zero_distance')
        self.assertEqual(stop_reason(0.,0.,0.,16),'near_zero_distance')
        self.assertEqual(stop_reason(100.,0.,0.,16),'sufficient_improvement')
        self.assertLess(improvement(100.,121.),0)

    def test_maximum_stops_even_if_all_candidates_predict_worsening(self):
        self.assertEqual(stop_reason(100.,121.,1.,48),'budget_exhausted')


if __name__=='__main__':unittest.main()
