"""G9 conditional warm start and G10 policy-initialized iCEM, protocol search-r4.

References: https://arxiv.org/abs/2008.06389 and
https://github.com/martius-lab/iCEM/blob/main/icem/controllers/icem.py
This task-specific Torch implementation uses the DESIGN.md 48/32/16 budget,
conditional policy tails, projected actions, and a global evaluated-best safeguard.
"""
import math
import torch


def top(costs, count=8):
    return torch.argsort(costs, stable=True)[:count]


def fit(mean, sigma, elites):
    return (.1*mean+.9*elites.float().mean(0),
            (.1*sigma+.9*elites.float().std(0,unbiased=False)).clamp_min(.05))


def colored_noise(count, generator, device, beta=2.):
    """Independent channels; 80-control-step FFT axis, zero DC, ensemble variance 1.

    Nonzero-bin expected power is f**(-beta). Normalization is deterministic,
    not sample-wise, so this remains Gaussian. No external RNG is used.
    """
    length=80
    freq=torch.fft.rfftfreq(length,device=device)
    scale=freq.clamp_min(1/length).pow(-beta/2)
    scale[0]=0
    real=torch.randn(count,25,len(freq),generator=generator,device=device)*scale
    imag=torch.randn(count,25,len(freq),generator=generator,device=device)*scale
    imag[...,0]=imag[...,-1]=0
    real[...,-1]*=math.sqrt(2.)
    # E[||irfft||^2]/length = (4*sum(interior scale^2)+2*Nyquist^2)/length^2.
    std=torch.sqrt(4*scale[1:-1].square().sum()+2*scale[-1].square())/length
    noise=torch.fft.irfft(torch.complex(real,imag),n=length,dim=-1)/std
    return noise.transpose(1,2).reshape(count,16,125)


class SearchPlanner:
    def __init__(self,mode):
        if mode not in ('warm_start','icem'):raise ValueError('unknown search mode')
        self.mode=mode
        self.reset()

    def reset(self):
        self.elites=self.mean=None

    def plan(self,agent,pctx,goal):
        icem=self.mode=='icem'
        policy_calls=[];wm_calls=[]
        frame_ctx=torch.stack(agent.frames)
        past=agent.executed[len(agent.executed)-len(agent.frames)+1:] if len(agent.frames)>1 else []
        past=torch.stack(past) if past else None
        def project(actions):
            if not icem:return actions
            a_mean=torch.as_tensor(agent.a_mean,device=agent.device)
            a_std=torch.as_tensor(agent.a_std,device=agent.device)
            return actions.maximum((-1-a_mean)/a_std).minimum((1-a_mean)/a_std)
        def sample(context,n,steps,purpose):
            policy_calls.append(dict(purpose=purpose,candidates=n,blocks=steps,action_blocks=n*steps))
            return project(agent.policy.sample(context,steps,agent.n_steps,agent.lag,agent.generator))
        def rollout(frames,actions,steps,purpose):
            wm_calls.append(dict(purpose=purpose,candidates=len(frames),steps=steps,latent_predictions=len(frames)*steps))
            with agent._autocast():
                return agent.encoder.rollout(frames,agent.encoder.action_encoder(actions),steps)
        def from_real(actions,purpose):
            n=len(actions)
            acts=torch.cat([past[None].expand(n,-1,-1),actions],1) if past is not None else actions
            return rollout(frame_ctx[None].expand(n,-1,-1),acts,actions.shape[1],purpose)
        cache=self.elites
        if cache is not None and (cache.ndim!=3 or cache.shape[1:]!=(16,125)):cache=None
        if cache is not None:
            cache=cache[torch.isfinite(cache).flatten(1).all(-1)][:8]
        reused=0 if cache is None else len(cache)
        tail=None
        if reused:
            prefix=cache[:,1:]
            pred_prefix=from_real(prefix,'warm_prefix')
            endpoint=((pred_prefix[:,-1].float()-agent.z_mean)/agent.z_std)[:,None]
            tail=sample(endpoint,reused,1,'conditional_tail')
            warm=torch.cat([prefix,tail],1)
            # z13,z14,z15 must be paired with u13,u14,new_tail (context=3).
            tail_frames=torch.cat([frame_ctx[None].expand(reused,-1,-1),pred_prefix],1)[:,-agent.encoder.cfg.context:]
            count=tail_frames.shape[1]-1
            tail_actions=torch.cat([prefix[:,-count:] if count else prefix[:,:0],tail],1)
            pred_tail=rollout(tail_frames,tail_actions,1,'conditional_tail')
            warm_pred=torch.cat([pred_prefix,pred_tail],1)
        fresh=48-reused
        blocks=sample(torch.stack(pctx)[None].expand(fresh,-1,-1),fresh,16,'fresh_policy')
        pred=from_real(blocks,'fresh_policy')
        if reused:
            blocks=torch.cat([blocks,warm]);pred=torch.cat([pred,warm_pred])
        costs=agent._candidate_cost(pred,goal)
        sources=['policy']*fresh+['warm_start']*reused
        rounds=[0]*48
        elite_ids=top(costs)
        traces=[dict(round=0,new_evaluations=48,reused_ids=[],elite_ids=elite_ids.cpu().tolist(),
                     round_best_cost=float(costs.min()),global_best_cost=float(costs.min()))]
        mean_origin=None
        if icem:
            if reused and self.mean is not None and torch.isfinite(self.mean).all():
                mean=project(torch.cat([self.mean[1:],tail.float().mean(0)],0))
                mean_origin='shifted_optimized_mean_with_conditioned_tail'
            else:
                mean=blocks.float().mean(0)
                mean_origin='initial_pool_mean'
            sigma=blocks.float().std(0,unbiased=False).clamp_min(.05)
            initial_sigma=dict(min=float(sigma.min()),mean=float(sigma.mean()),max=float(sigma.max()))
            for iteration,new_count in ((1,32),(2,16)):
                mean,sigma=fit(mean,sigma,blocks[elite_ids])
                carried=elite_ids[:2]
                noises=colored_noise(new_count if iteration==1 else new_count-1,agent.generator,agent.device)
                proposals=project(mean[None]+sigma[None]*noises)
                if iteration==2:proposals=torch.cat([project(mean)[None],proposals])
                new_pred=from_real(proposals,f'icem_round_{iteration}')
                new_cost=agent._candidate_cost(new_pred,goal)
                start=len(costs)
                blocks=torch.cat([blocks,proposals]);pred=torch.cat([pred,new_pred]);costs=torch.cat([costs,new_cost])
                sources+=(['mean']+['cem']*15) if iteration==2 else ['cem']*32
                rounds += [iteration]*new_count
                available=torch.cat([carried,torch.arange(start,start+new_count,device=costs.device)])
                # Stable IDs settle exact cost ties even across the retained/new pools.
                available=available.sort().values
                elite_ids=available[top(costs[available])]
                traces.append(dict(round=iteration,new_evaluations=new_count,reused_ids=carried.cpu().tolist(),
                    elite_ids=elite_ids.cpu().tolist(),round_best_cost=float(costs[available].min()),
                    global_best_cost=float(costs.min()),sigma_min=float(sigma.min()),sigma_mean=float(sigma.mean()),
                    sigma_max=float(sigma.max())))
            # Like iCEM, fit after final elite selection; cache this mean only, never execute it unscored.
            mean,sigma=fit(mean,sigma,blocks[elite_ids])
            self.mean=project(mean).detach().clone()
        best=int(costs.argmin())
        global_elites=top(costs)
        self.elites=blocks[global_elites].detach().clone()
        agent.last_distance=costs
        info=dict(sampling_mode=self.mode,actual_candidates=len(costs),trajectory_evaluations=len(costs),
            sampling_batch_size=48,max_candidates=96 if icem else 48,min_improvement=None,
            policy_sample_calls=len(policy_calls),wm_rollout_calls=len(wm_calls),
            wm_predict_calls=sum(c['steps'] for c in wm_calls),
            policy_denoising_calls=sum(agent.n_steps+(c['blocks']-1)*agent.lag for c in policy_calls),
            policy_action_blocks=sum(c['action_blocks'] for c in policy_calls),
            future_latent_predictions=sum(c['latent_predictions'] for c in wm_calls),
            policy_sample_batches=policy_calls,wm_rollout_batches=wm_calls,
            candidate_ids=list(range(len(costs))),candidate_sources=sources,candidate_rounds=rounds,
            selected_source=sources[best],selected_round=rounds[best],selected_batch=rounds[best]+1,
            fresh_policy_candidates=fresh,valid_cached_candidates=reused,conditional_tail_blocks=reused,
            global_elite_ids=global_elites.cpu().tolist(),search_rounds=traces,
            cumulative_best_costs=[t['global_best_cost'] for t in traces],
            batch_best_costs=[t['round_best_cost'] for t in traces],stop_reason='fixed_search_budget')
        if icem:info.update(initial_sigma=initial_sigma,mean_initialization=mean_origin,
                            mean_candidate_id=80,final_sigma_min=float(sigma.min()),
                            final_sigma_mean=float(sigma.mean()),final_sigma_max=float(sigma.max()),
                            projection='normalized raw-action bounds; all evaluated/cached plans projected',
                            colored_noise=dict(beta=2.,time_axis=80,channels=25,dc='zero',normalization='deterministic ensemble variance'))
        return blocks,pred,best,info
