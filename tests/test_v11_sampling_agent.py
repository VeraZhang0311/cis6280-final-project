import unittest
import numpy as np
import torch
from assignment.training.planning.agent import GoalPlanningAgent
from test_planning_diagnostics import fake_agent


def controlled(costs, diagnostics=True, adaptive=True, candidates=16):
    agent=fake_agent(GoalPlanningAgent,diagnostics)
    agent.scoring='tail_min'
    agent.goals=torch.tensor([[10.,0.]])
    agent.gap=torch.ones(1)
    agent.adaptive_sampling=adaptive
    agent.max_candidates=48
    agent.min_improvement=.20
    agent.candidates=candidates
    calls=[]
    def sample(context,*args):
        torch.rand(1,generator=args[-1])
        calls.append(dict(context=context.clone(),frames=None,acts=None))
        return torch.full((len(context),16,125),len(calls)/10.)
    def rollout(frames,acts,horizon):
        agent.encoder.rollout_calls+=1
        calls[-1].update(frames=frames.clone(),acts=acts.clone())
        pred=torch.zeros(len(frames),16,2)
        pred[:,:,0]=10.
        pred[:,:,1]=float(costs[len(calls)-1])**.5
        return pred
    agent.policy.sample=sample
    agent.encoder.rollout=rollout
    return agent,calls


class SamplingAgent(unittest.TestCase):
    def test_one_batch_when_sufficient_improvement(self):
        agent,calls=controlled([64.])
        agent.step(np.zeros((2,2,3),np.uint8))
        self.assertEqual(len(calls),1)
        self.assertEqual(agent.diagnostic['actual_candidates'],16)
        self.assertEqual(agent.diagnostic['stop_reason'],'sufficient_improvement')

    def test_second_batch_improves_and_third_is_not_sampled(self):
        agent,calls=controlled([81.,64.])
        np.testing.assert_allclose(agent.step(np.zeros((2,2,3),np.uint8)),.2)
        self.assertEqual(len(calls),2)
        self.assertEqual(agent.diagnostic['actual_candidates'],32)
        self.assertEqual(agent.diagnostic['selected_candidate'],16)
        self.assertEqual(agent.diagnostic['selected_batch'],2)
        self.assertEqual(len(agent.diagnostic['candidate_scores']),32)
        self.assertEqual(len(agent.frames),1)
        self.assertEqual(len(agent.history),1)
        self.assertEqual(len(agent.executed),1)
        torch.testing.assert_close(calls[0]['context'],calls[1]['context'])
        torch.testing.assert_close(calls[0]['frames'],calls[1]['frames'])
        self.assertEqual(agent.index,0)  # imagined matches never advance actual goal

    def test_retains_earlier_best_even_if_later_batches_worse_or_tied(self):
        for costs in ([121.,144.,169.],[81.,81.,81.]):
            agent,calls=controlled(costs)
            np.testing.assert_allclose(agent.step(np.zeros((2,2,3),np.uint8)),.1)
            self.assertEqual(len(calls),3)
            self.assertEqual(agent.diagnostic['actual_candidates'],48)
            self.assertEqual(agent.diagnostic['selected_candidate'],0)
            self.assertEqual(agent.diagnostic['selected_batch'],1)
            self.assertEqual(agent.diagnostic['stop_reason'],'budget_exhausted')
            self.assertEqual(agent.diagnostic['policy_sample_calls'],3)
            self.assertEqual(agent.diagnostic['wm_rollout_calls'],3)

    def test_context_includes_executed_actions_once_and_logs_do_not_consume_rng(self):
        plain,a=controlled([81.,81.,81.,81.,81.,81.],False)
        logged,b=controlled([81.,81.,81.,81.,81.,81.],True)
        for _ in range(2):
            frame=np.zeros((2,2,3),np.uint8)
            np.testing.assert_array_equal(plain.step(frame),logged.step(frame))
            self.assertTrue(torch.equal(plain.generator.get_state(),logged.generator.get_state()))
        self.assertEqual(len(logged.executed),2)
        self.assertEqual(b[3]['frames'].shape[1],2)
        for call in b[3:]:
            self.assertEqual(call['acts'].shape[1],17)
            torch.testing.assert_close(call['acts'][:,:1],b[3]['acts'][:,:1])
            torch.testing.assert_close(call['frames'],b[3]['frames'])

    def test_g7_has_one_48_candidate_policy_and_rollout_batch(self):
        agent,calls=controlled([81.],adaptive=False,candidates=48)
        agent.step(np.zeros((2,2,3),np.uint8))
        self.assertEqual(len(calls),1)
        self.assertEqual(len(calls[0]['context']),48)
        self.assertEqual(agent.encoder.rollout_calls,1)
        self.assertEqual(agent.diagnostic['actual_candidates'],48)
        self.assertEqual(agent.diagnostic['policy_sample_calls'],1)
        self.assertEqual(agent.diagnostic['wm_rollout_calls'],1)
        self.assertEqual(agent.diagnostic['stop_reason'],'fixed_budget')


if __name__=='__main__':unittest.main()
