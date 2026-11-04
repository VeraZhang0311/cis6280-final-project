import unittest

from assignment.training.planning.alignment import GoalRealigner


class AlignmentRules(unittest.TestCase):
    def test_default_dwell_window_and_three_real_confirmations(self):
        align = GoalRealigner()
        calls = []
        def future():
            calls.append(True)
            return [dict(index=1, distance_sq=.2, threshold_sq=1.)]
        for _ in range(48):
            matched, info = align.observe(0, 10., future)
            self.assertIsNone(matched)
        self.assertFalse(calls)
        for count in (1, 2, 3):
            matched, info = align.observe(0, 10., future)
            self.assertTrue(info['stagnation_condition_met'])
            self.assertEqual(info['pending_match_count'], count)
            self.assertEqual(matched, 1 if count == 3 else None)

    def test_improvement_or_official_switch_clears_pending(self):
        align = GoalRealigner()
        supplier = lambda: [dict(index=1, distance_sq=.2, threshold_sq=1.)]
        for _ in range(50):
            align.observe(0, 10., supplier)
        self.assertEqual(align.count, 2)
        matched, info = align.observe(1, 7., lambda: self.fail('new goal must restart dwell'))
        self.assertIsNone(matched)
        self.assertEqual(align.count, 0)
        self.assertFalse(info['stagnation_condition_met'])
        align.reset()
        for k in range(49):
            matched, info = align.observe(0, 20.-k*.2, lambda: self.fail('distance is still improving'))
        self.assertFalse(info['stagnation_condition_met'])
        self.assertGreaterEqual(info['stagnation_relative_improvement'], .10)

    def test_matching_uses_threshold_ratio_ties_and_rejects_zero_threshold(self):
        align = GoalRealigner()
        choices = [dict(index=3, distance_sq=0., threshold_sq=0.),
                   dict(index=2, distance_sq=.4, threshold_sq=2.),
                   dict(index=1, distance_sq=.2, threshold_sq=1.)]
        for _ in range(51):
            matched, info = align.observe(0, 10., lambda: choices)
        self.assertEqual(matched, 1)
        self.assertEqual(info['matched_confirmations'], 3)
        self.assertEqual(info['matched_goal_q'], .2)
        align.changed(2, 10.)
        with self.assertRaises(ValueError):
            align.observe(1, 10., lambda: [])
        align.reset()
        self.assertEqual(align.tick, -1)
        self.assertIsNone(align.pending)
        self.assertFalse(align.history)

    def test_nonmatching_real_observation_breaks_confirmation(self):
        align = GoalRealigner()
        match = lambda: [dict(index=1, distance_sq=.1, threshold_sq=1.)]
        for _ in range(50):
            align.observe(0, 10., match)
        self.assertEqual(align.count, 2)
        align.observe(0, 10., lambda: [])
        self.assertEqual(align.count, 0)
        for _ in range(2):
            matched, _ = align.observe(0, 10., match)
            self.assertIsNone(matched)
        self.assertEqual(align.observe(0, 10., match)[0], 1)


class BidirectionalRules(unittest.TestCase):
    def ready(self, index=4):
        align = GoalRealigner(bidirectional=True)
        for _ in range(48):
            align.observe(index, 10., lambda: self.fail('dwell not yet met'))
        return align

    def test_backward_and_forward_with_three_confirmations(self):
        for target in (1, 7):
            align = self.ready()
            supplier = lambda: [dict(index=target, distance_sq=.1, threshold_sq=1.)]
            for count in (1, 2, 3):
                match, info = align.observe(4, 10., supplier)
                self.assertEqual(match, target if count==3 else None)
            align.changed(target, .1)
            self.assertIsNone(align.pending)
            self.assertFalse(align.observe(target, .1, lambda: self.fail('reset dwell'))[1]['stagnation_condition_met'])
            align.reset()
            self.assertEqual(align.tick, -1)

    def test_current_nearest_change_then_earlier_tie_breaks(self):
        def candidate(index, q):
            return dict(index=index, distance_sq=q, threshold_sq=1.)
        align = self.ready()
        self.assertIsNone(align.observe(4, 10., lambda: [candidate(3,.1),candidate(4,.1)])[0])
        self.assertIsNone(align.pending)
        align.observe(4, 10., lambda: [candidate(1,.1),candidate(5,.1)])
        self.assertEqual(align.pending, 5)
        align.observe(4, 10., lambda: [candidate(3,.1),candidate(5,.1)])
        self.assertEqual(align.pending, 3)
        self.assertEqual(align.count, 1)
        # A current-target best match cancels a pending change.
        align.observe(4, 10., lambda: [candidate(3,.1),candidate(4,.05)])
        self.assertIsNone(align.pending)

    def test_timeout_without_match_and_zero_threshold_never_moves(self):
        align = self.ready()
        for _ in range(100):
            match, info = align.observe(4, 10., lambda: [dict(index=1,distance_sq=0.,threshold_sq=0.),
                                                       dict(index=5,distance_sq=1.,threshold_sq=1.)])
            self.assertIsNone(match)
            self.assertFalse(info['nearby_match_found'])
        with self.assertRaises(ValueError):
            align.observe(4, 10., lambda: [dict(index=0,distance_sq=0.,threshold_sq=1.)])
        # Window is centered on current, not historical maximum; revisiting is permitted.
        align.changed(1, 10.)
        for _ in range(51):
            match, _ = align.observe(1, 10., lambda: [dict(index=0,distance_sq=.1,threshold_sq=1.)])
        self.assertEqual(match, 0)


if __name__ == '__main__':
    unittest.main()
