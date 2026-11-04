import numpy as np
import torch
import unittest

from assignment.training.planning.agent import GoalPlanningAgent
from assignment.training.planning.alignment import GoalRealigner
from assignment.training.data.cache import STRIDE, RAW_ACTION_DIM
from test_planning_diagnostics import fake_agent


@torch.no_grad()
def legacy_step(agent, frame):
    """Unmodified v1.0 G0 step, retained independently for action/RNG/call equivalence."""
    z = agent._encode(frame[None])[0]
    agent.frames = (agent.frames+[z])[-agent.encoder.cfg.context:]
    agent.history = (agent.history+[(z-agent.z_mean)/agent.z_std])[-agent.policy.cfg.context:]
    goal = agent.goals[agent.goal_index(z)]
    pctx = [agent.history[0]]*(agent.policy.cfg.context-len(agent.history))+agent.history
    n = agent.candidates
    blocks = agent.policy.sample(torch.stack(pctx)[None].expand(n, -1, -1), agent.generate,
                                agent.n_steps, agent.lag, agent.generator)
    past = agent.executed[len(agent.executed)-len(agent.frames)+1:] if len(agent.frames)>1 else []
    acts = torch.cat([torch.stack(past)[None].expand(n, -1, -1), blocks], 1) if past else blocks
    with agent._autocast():
        pred = agent.encoder.rollout(torch.stack(agent.frames)[None].expand(n, -1, -1),
                                     agent.encoder.action_encoder(acts), agent.generate)
    agent.last_distance = (pred[:, -1].float()-goal).square().sum(-1)
    best = int(agent.last_distance.argmin())
    raw = np.clip(blocks[best, :agent.execute].cpu().numpy()*agent.a_std+agent.a_mean, -1, 1)
    block = raw.reshape(-1, STRIDE, RAW_ACTION_DIM)[0]
    agent.executed.append(torch.from_numpy((block.reshape(-1)-agent.a_mean)/agent.a_std).to(agent.device))
    return block


class VariantAgent(unittest.TestCase):
    def test_g0_matches_legacy_actions_rng_goal_and_rollout_calls(self):
        reference, current = fake_agent(GoalPlanningAgent, False), fake_agent(GoalPlanningAgent, True)
        current.scoring = 'terminal'
        current.realigner = None
        for value in (0, 20, 40, 80, 120, 160):
            frame = np.full((2, 2, 3), value, np.uint8)
            np.testing.assert_array_equal(legacy_step(reference, frame), current.step(frame))
            self.assertTrue(torch.equal(reference.generator.get_state(), current.generator.get_state()))
            self.assertEqual(reference.index, current.index)
            self.assertEqual(reference.encoder.rollout_calls, current.encoder.rollout_calls)
            torch.testing.assert_close(reference.last_distance, current.last_distance, rtol=0, atol=0)

    def test_tail_formulas_select_different_candidates(self):
        agent = GoalPlanningAgent.__new__(GoalPlanningAgent)
        pred = torch.zeros(3, 16, 2)
        pred[:, -4:, 0] = torch.sqrt(torch.tensor([[9., 9., 9., 1.], [0., 20., 20., 2.], [3., 3., 3., 3.]]))
        for mode, costs, chosen in [('terminal', [1., 2., 3.], 0),
                                     ('tail_min', [1., 0., 3.], 1),
                                     ('tail_weighted', [5.8, 10.8, 3.], 2)]:
            agent.scoring = mode
            actual = agent._candidate_cost(pred, torch.zeros(2))
            np.testing.assert_allclose(actual.numpy(), costs, rtol=1e-6)
            self.assertEqual(int(actual.argmin()), chosen)

    def test_tail_min_executes_first_block_not_closest_future_block(self):
        agent = fake_agent(GoalPlanningAgent, True)
        agent.scoring, agent.candidates = 'tail_min', 3
        agent.goals, agent.gap = torch.zeros(1, 2), torch.ones(1)
        blocks = torch.ones(3, 16, 125)*.1
        blocks[1, :, :] = .2
        blocks[1, 12, :] = .7
        blocks[2, :, :] = .3
        pred = torch.zeros(3, 16, 2)
        pred[:, -4:, 0] = torch.sqrt(torch.tensor([[9., 9., 9., 1.], [0., 20., 20., 2.], [3., 3., 3., 3.]]))
        agent.policy.sample = lambda *args: blocks
        agent.encoder.rollout = lambda *args: pred
        np.testing.assert_allclose(agent.step(np.zeros((2, 2, 3), np.uint8)), .2)
        self.assertEqual(agent.diagnostic['selected_candidate'], 1)
        self.assertEqual(agent.diagnostic['selected_tail_closest_block'], 13)

    def test_realignment_then_official_advance_uses_real_observations_and_resets(self):
        agent = fake_agent(GoalPlanningAgent, True)
        agent.goals = torch.tensor([[10., 10.], [0., 0.], [20., 20.]])
        agent.gap = torch.ones(3)
        agent.realigner = GoalRealigner(dict(min_dwell_s=0., window_s=3.5))
        frame = np.zeros((2, 2, 3), np.uint8)
        for _ in range(9):
            agent.step(frame)
            self.assertEqual(agent.index, 0)
        agent.step(frame)
        self.assertEqual(agent.index, 2)
        self.assertEqual(agent.diagnostic['goal_change_reason'], 'realignment_then_official')
        self.assertEqual(agent.diagnostic['matched_goal_index'], 1)
        self.assertEqual(agent.diagnostic['matched_confirmations'], 3)
        self.assertEqual(agent.diagnostic['goal_distance_sq'], 800.)
        agent.reset()
        self.assertEqual(agent.realigner.tick, -1)
        self.assertIsNone(agent.realigner.pending)

    def test_bidirectional_uses_matched_target_this_call_then_official_next_call(self):
        for origin, target in ((3, 1), (0, 2)):
            agent = fake_agent(GoalPlanningAgent, True)
            agent.goals = torch.tensor([[10.,10.],[0.,0.],[20.,20.],[30.,30.]]) if target==1 else torch.tensor([[10.,10.],[20.,20.],[0.,0.],[30.,30.]])
            agent.gap = torch.ones(4)
            agent.index = origin
            agent.realigner = GoalRealigner(dict(min_dwell_s=0.,window_s=3.5), bidirectional=True)
            frame = np.zeros((2,2,3),np.uint8)
            for _ in range(9):
                agent.step(frame)
                self.assertEqual(agent.index, origin)
            agent.step(frame)
            self.assertEqual(agent.index, target)  # no same-call while undoing the match
            self.assertEqual(agent.diagnostic['goal_distance_sq'], 0.)
            self.assertEqual(agent.diagnostic['goal_change_direction'], 'backward' if target<origin else 'forward')
            self.assertEqual([c['index'] for c in agent.diagnostic['alignment_goal_checks']], list(range(4)))
            agent.step(frame)
            self.assertEqual(agent.index, target+1)
            self.assertFalse(agent.diagnostic['stagnation_condition_met'])

    def test_bidirectional_search_clips_both_boundaries(self):
        for index in (0, 5):
            agent = fake_agent(GoalPlanningAgent, True)
            agent.goals = torch.tensor([[10.+i,10.+i] for i in range(6)])
            agent.gap = torch.ones(6)
            agent.index = index
            agent.realigner = GoalRealigner(dict(min_dwell_s=0.,window_s=3.5), bidirectional=True)
            for _ in range(8):
                agent.step(np.zeros((2,2,3),np.uint8))
            self.assertEqual([c['index'] for c in agent.diagnostic['alignment_goal_checks']],
                             list(range(max(0,index-3), min(6,index+4))))


if __name__ == '__main__':
    unittest.main()
