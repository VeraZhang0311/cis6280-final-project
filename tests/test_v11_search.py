import unittest
import numpy as np
import torch
from assignment.training.planning.agent import GoalPlanningAgent
from assignment.training.planning.search import SearchPlanner, fit, colored_noise
from test_planning_diagnostics import fake_agent


def make(mode=None):
    a=fake_agent(GoalPlanningAgent,True)
    a.scoring='tail_min';a.candidates=48
    a.goals=torch.tensor([[1000.,1000.]])
    a.gap=torch.ones(1)
    a.search=SearchPlanner(mode) if mode else None
    return a


def spy(a):
    samples=[];rollouts=[]
    sample=a.policy.sample;rollout=a.encoder.rollout
    def p(ctx,steps,*args):
        actions=sample(ctx,steps,*args)
        samples.append(dict(ctx=ctx.clone(),steps=steps,actions=actions.clone()))
        return actions
    def wm(frames,acts,steps):
        pred=rollout(frames,acts,steps)
        rollouts.append(dict(frames=frames.clone(),actions=acts.clone(),steps=steps,pred=pred.clone()))
        return pred
    a.policy.sample=p;a.encoder.rollout=wm
    return samples,rollouts


class WarmStart(unittest.TestCase):
    def test_cold_g9_exactly_matches_g7_actions_rng_and_calls(self):
        fixed,warm=make(),make('warm_start')
        frame=np.zeros((2,2,3),np.uint8)
        np.testing.assert_array_equal(fixed.step(frame),warm.step(frame))
        self.assertTrue(torch.equal(fixed.generator.get_state(),warm.generator.get_state()))
        self.assertEqual(warm.encoder.rollout_calls,1)
        torch.testing.assert_close(warm.last_distance,fixed.last_distance,atol=0,rtol=0)
        self.assertEqual(warm.diagnostic['valid_cached_candidates'],0)
        self.assertEqual(len(warm.search.elites),8)

    def test_fresh_origin_prefix_endpoint_policy_and_tail_action_alignment(self):
        a=make('warm_start');samples,calls=spy(a)
        a.step(np.zeros((2,2,3),np.uint8))
        old_elites=a.search.elites.clone()
        executed=a.executed[-1].clone()
        a.z_mean=torch.tensor([.1,.2]);a.z_std=torch.tensor([2.,3.])
        a.step(np.full((2,2,3),64,np.uint8))
        prefix,tail,fresh=calls[1:]
        self.assertEqual([r['steps'] for r in calls],[16,15,1,16])
        self.assertEqual([s['steps'] for s in samples],[16,1,16])
        torch.testing.assert_close(prefix['actions'][:,1:],old_elites[:,1:])
        torch.testing.assert_close(prefix['actions'][:,0],executed.expand(8,-1))
        self.assertTrue(torch.allclose(prefix['frames'][:,-1],torch.full((8,2),64/255)))
        torch.testing.assert_close(samples[1]['ctx'][:,0],(prefix['pred'][:,-1]-a.z_mean)/a.z_std)
        torch.testing.assert_close(tail['frames'],prefix['pred'][:,-3:])
        torch.testing.assert_close(tail['actions'][:,:2],old_elites[:,-2:])
        torch.testing.assert_close(tail['actions'][:,-1:],samples[1]['actions'])
        self.assertEqual((len(a.frames),len(a.history),len(a.executed)),(2,1,2))
        self.assertEqual(a.diagnostic['trajectory_evaluations'],48)
        self.assertEqual(a.diagnostic['future_latent_predictions'],48*16)
        self.assertEqual(a.diagnostic['policy_action_blocks'],40*16+8)
        self.assertEqual(a.diagnostic['policy_sample_calls'],2)
        self.assertEqual(a.diagnostic['wm_rollout_calls'],3)

    def test_partial_cache_and_new_target_are_rescored_then_reset(self):
        a=make('warm_start');a.step(np.zeros((2,2,3),np.uint8))
        a.search.elites=a.search.elites[:5]
        a.goals=torch.tensor([[0.,0.],[20.,20.]])
        a.gap=torch.ones(2)
        a.step(np.zeros((2,2,3),np.uint8))
        self.assertEqual(a.index,1)
        self.assertEqual(a.diagnostic['valid_cached_candidates'],5)
        self.assertEqual(a.diagnostic['fresh_policy_candidates'],43)
        self.assertEqual(len(a.last_distance),48)
        self.assertEqual(a.diagnostic['future_latent_predictions'],768)
        a.reset()
        self.assertIsNone(a.search.elites)
        self.assertIsNone(a.search.mean)


class ICEM(unittest.TestCase):
    def test_float32_momentum_and_std_floor(self):
        mu,sigma=fit(torch.full((16,125),10.),torch.full((16,125),2.),torch.full((8,16,125),3.))
        torch.testing.assert_close(mu,torch.full_like(mu,3.7))
        torch.testing.assert_close(sigma,torch.full_like(sigma,.2))
        _,sigma=fit(mu,torch.zeros_like(sigma),torch.ones(8,16,125))
        self.assertTrue(torch.equal(sigma,torch.full_like(sigma,.05)))

    def test_noise_control_time_axis_psd_and_seed(self):
        first=colored_noise(256,torch.Generator().manual_seed(42),'cpu')
        second=colored_noise(256,torch.Generator().manual_seed(42),'cpu')
        self.assertTrue(torch.equal(first,second))
        self.assertEqual(first.shape,(256,16,125))
        spectrum=torch.fft.rfft(first.reshape(256,80,25),dim=1).abs().square().mean((0,2))
        self.assertGreater(float(spectrum[1]/spectrum[2]),3.4)
        self.assertLess(float(spectrum[1]/spectrum[2]),4.6)
        self.assertLess(float(spectrum[0]),1e-7)
        self.assertAlmostEqual(float(first.square().mean()),1.,delta=.08)

    def test_budget_mean_slot_global_best_and_global_elite_cache(self):
        a=make('icem');samples,calls=spy(a)
        evaluated=[]
        def costs(pred,goal):
            start=(0,100,200)[len(evaluated)]
            c=torch.arange(len(pred),dtype=torch.float32)+start
            evaluated.append(c)
            return c
        a._candidate_cost=costs
        block=a.step(np.zeros((2,2,3),np.uint8))
        info=a.diagnostic
        self.assertEqual([len(r['frames']) for r in calls],[48,32,16])
        self.assertEqual([r['new_evaluations'] for r in info['search_rounds']],[48,32,16])
        self.assertEqual([len(r['reused_ids']) for r in info['search_rounds']],[0,2,2])
        self.assertEqual(info['trajectory_evaluations'],96)
        self.assertEqual(info['future_latent_predictions'],96*16)
        self.assertEqual(info['candidate_sources'][80],'mean')
        self.assertEqual(info['selected_candidate'],0)
        self.assertEqual(info['selected_source'],'policy')
        self.assertEqual(info['global_elite_ids'],list(range(8)))
        np.testing.assert_array_equal(block,samples[0]['actions'][0,0].clamp(-1,1).reshape(5,25).numpy())
        torch.testing.assert_close(a.search.elites,samples[0]['actions'][:8].clamp(-1,1))
        self.assertEqual(len(a.frames),1)
        a.reset();self.assertIsNone(a.search.mean)

    def test_warm_budget_projection_and_sigma_reinitialization(self):
        a=make('icem');a.a_mean=np.full(125,.2,np.float32);a.a_std=np.full(125,2.,np.float32)
        a.policy.sample=lambda ctx,steps,*args:torch.full((len(ctx),steps,125),100.)
        _,calls=spy(a)
        a.step(np.zeros((2,2,3),np.uint8))
        a.step(np.zeros((2,2,3),np.uint8))
        info=a.diagnostic
        self.assertEqual(info['future_latent_predictions'],1536)
        self.assertEqual(info['trajectory_evaluations'],96)
        self.assertEqual((info['policy_sample_calls'],info['wm_rollout_calls']),(2,5))
        self.assertEqual(info['policy_action_blocks'],648)
        self.assertEqual(info['mean_initialization'],'shifted_optimized_mean_with_conditioned_tail')
        self.assertGreaterEqual(info['initial_sigma']['min'],.05-1e-7)
        for call in calls:
            raw=call['actions']*2+.2
            self.assertLessEqual(float(raw.max()),1.+1e-6)
            self.assertGreaterEqual(float(raw.min()),-1.-1e-6)


class FixedBudgets(unittest.TestCase):
    def test_32_and_96_single_batch(self):
        for n in (32,96):
            a=make();a.candidates=n;samples,calls=spy(a)
            a.step(np.zeros((2,2,3),np.uint8))
            self.assertEqual((len(samples),len(calls)),(1,1))
            self.assertEqual(a.diagnostic['trajectory_evaluations'],n)
            self.assertEqual(a.diagnostic['future_latent_predictions'],16*n)
            self.assertEqual(len(a.diagnostic['candidate_scores']),n)


if __name__=='__main__':unittest.main()
