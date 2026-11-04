"""Fixed full G8/G9/G10 development comparison, followed by predeclared new-scene validation."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np

from assignment.training.planning.diagnostics import file_identity, write_json
from assignment.training.planning.phase1 import init_phase1_worker, source_identity, validate_goals
from assignment.training.planning.v11 import TASK, SEEDS, variants, digest, paired_comparison, run_variant
from assignment.training.planning.v11_sampling import reference_metadata

VERSION='v1.1-search-r4'
ICEM_SOURCE='98c1c1fe2bfc94a87e4658b6c793f4e81bc30203'


def search_variants():
    base=dict(variants()['G1'],candidates=48)
    return {'G8-32':dict(base,candidates=32),'G8-96':dict(base,candidates=96),
            'G9':dict(base,search_mode='warm_start'),'G10':dict(base,search_mode='icem')}


def choose_new_method(full):
    order=list(search_variants())
    winner=min(order,key=lambda g:(-full[g]['mean_max_placed'],-full[g]['mean_final_count'],
                                   full[g]['decision_s_mean'],order.index(g)))
    return dict(variant=winner,config=search_variants()[winner],
                rule='max development mean peak; then max mean final; then min mean decision seconds; then fixed G8-32/G8-96/G9/G10 order',
                scope='Only development full outcomes select the method; heldout outcomes never change configuration')


def records(folder):
    return [json.loads((p.parent/'summary.json').read_text()) for p in sorted((folder/'episodes').glob('*/complete.json'))]


def compare(base, treatment, base_name, treatment_name):
    return dict(paired_comparison(base,treatment),baseline=base_name,treatment=treatment_name)


def summarize_search(folder,stats):
    rows=[]
    for p in sorted((folder/'episodes').glob('*/complete.json')):
        rows.extend(json.loads(s) for s in (p.parent/'steps.jsonl').read_text().splitlines())
    def avg(key,fallback):return float(np.mean([r.get(key,fallback(r)) for r in rows]))
    count=lambda r:r.get('actual_candidates',16)
    info=dict(decisions=len(rows),mean_trajectory_evaluations=avg('trajectory_evaluations',count),
              mean_future_latent_predictions=avg('future_latent_predictions',lambda r:count(r)*16),
              mean_policy_calls=avg('policy_sample_calls',lambda r:1),
              mean_wm_calls=avg('wm_rollout_calls',lambda r:1),
              mean_wm_predict_calls=avg('wm_predict_calls',lambda r:16),
              mean_policy_denoising_calls=avg('policy_denoising_calls',lambda r:10),
              mean_policy_action_blocks=avg('policy_action_blocks',lambda r:count(r)*16),
              mean_valid_cached_candidates=avg('valid_cached_candidates',lambda r:0),
              mean_conditional_tail_blocks=avg('conditional_tail_blocks',lambda r:0),
              mean_fresh_policy_candidates=avg('fresh_policy_candidates',count),
              winner_sources=dict(Counter(r.get('selected_source','policy') for r in rows)),
              winner_rounds=dict(Counter(str(r.get('selected_round',0)) for r in rows)),
              cache_available_rate=float(np.mean([r.get('valid_cached_candidates',0)>0 for r in rows])),
              note='Old fixed G1/G7 missing counters inferred from their validated fixed-budget configuration')
    result=dict(stats);result['search']=info
    return result


def references(source,workers,gpus):
    parent,g1=reference_metadata(source,workers,gpus)
    root=Path(source).resolve();group=root/'full/G7'
    config=json.loads((group/'run.json').read_text())
    expected=dict(variants()['G1'],candidates=48)
    if config['config']!=expected or len(records(group))!=60:
        raise ValueError('compatible complete G7 is required')
    ids={(r['scene_seed'],r['sample_seed']) for r in records(group)}
    if ids!={(s,k) for s in parent['protocol']['full_scenes'] for k in SEEDS}:
        raise ValueError('reference G7 case identities differ')
    for p in (group/'episodes').glob('*/complete.json'):
        if json.loads(p.read_text())['protocol_id']!=config['protocol_id']:
            raise ValueError('reference G7 protocol mismatch')
    g7=dict(path=str(group),group_manifest=file_identity(group/'run.json'),summary=file_identity(group/'summary.json'),
            cases=[{f:file_identity(p.parent/f) for f in ('complete.json','summary.json','steps.jsonl','events.jsonl')}
                   for p in sorted((group/'episodes').glob('*/complete.json'))])
    return parent,dict(G1=g1,G7=g7)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--gpus',default='0,1')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    parent,reference=references(args.reference_run,args.workers,args.gpus)
    repo=Path(__file__).resolve().parents[3]
    old=parent['protocol'];scenes=old['full_scenes']
    holdout_file=repo/'notes/v1.1/validation-scenes-20-next.json'
    heldout=json.loads(holdout_file.read_text())
    inputs={k:file_identity(v['path']) for k,v in old['inputs'].items()}
    if any(inputs[k]['sha256']!=v['sha256'] for k,v in old['inputs'].items()):
        raise ValueError('frozen input bytes changed')
    with np.load(inputs['goals']['path']) as goals:
        expected=sorted(int(x) for x in goals['seeds'] if x>max(scenes))[:20]
    if heldout!=expected or len(heldout)!=20 or set(heldout)&set(scenes):
        raise ValueError('heldout must be the fixed next 20 official scenes')
    validate_goals(inputs['goals']['path'],scenes+heldout)
    code=source_identity()
    if (repo/'source-provenance.json').exists():code['snapshot_origin']=json.loads((repo/'source-provenance.json').read_text())
    protocol=dict(version=VERSION,base_variant='G7',scoring='tail_min',goal_selection='official',
        code=code,design=file_identity(repo/'notes/v1.1/DESIGN.md'),inputs=inputs,task=TASK,
        full_scenes=scenes,heldout_scenes=heldout,heldout_scene_file=file_identity(holdout_file),sample_seeds=SEEDS,
        workers=args.workers,gpus=args.gpus,pilot=False,pilot_override='User requested direct full experiments; prior no-pilot preference retained',
        variants=search_variants(),reference=reference,evidence=dict(videos='all episodes',tail_state_steps=56,keyframes=6),
        warm_start=dict(elites=8,fresh=40,prefix_steps=15,tail_steps=1,tail_policy_context='normalized newly imagined endpoint',
                        re_evaluate_from='current real context',cache_reset='episode reset',partial_cache='fill missing slots with fresh policy'),
        icem=dict(reference_paper='https://arxiv.org/abs/2008.06389',official_commit=ICEM_SOURCE,
                  official_code=f'https://github.com/martius-lab/iCEM/blob/{ICEM_SOURCE}/icem/controllers/icem.py',
                  implementation='Independent task-specific Torch search.py v1; policy-initialized, not paper-default parameters',
                  populations=[48,32,16],elite_count=8,within_round_reuse=2,beta=2.,noise_time_steps=80,
                  noise_channels=25,noise_dc='zero',noise_normalization='deterministic ensemble variance',alpha_old_weight=.1,
                  sigma_floor=.05,mean_candidate_id=80,projection='all evaluated/cached actions in normalized raw-action bounds',
                  distribution_fit='after each of three elite selections; final updated mean cached only'),
        selection_rule='max development mean peak; then mean final; then min mean decision seconds; then fixed variant order',
        stage_order=['G8-32','G8-96','G9','G10','heldout G1','heldout G7','heldout selected new method'])
    protocol_id=digest(protocol);root=args.out.resolve()
    if args.resume:
        manifest=json.loads((root/'run.json').read_text())
        if manifest['protocol_id']!=protocol_id:raise ValueError('resume source/settings/reference mismatch')
        manifest.update(status='running',resumed_utc=datetime.now(timezone.utc).isoformat())
    else:
        root.mkdir(parents=True,exist_ok=False)
        manifest=dict(schema_version=1,status='running',protocol_id=protocol_id,protocol=protocol,
                      runtime=parent['runtime'],execution_signature=parent['execution_signature'],
                      completed_new_episodes=0,reference_episodes=120,planned_new_episodes=420,
                      started_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'run.json',manifest)
    ref_root=root/'reference';ref_root.mkdir(exist_ok=True)
    report=dict(protocol_version=VERSION,pilot=False,full={},paired={},heldout={},heldout_paired={})
    baseline={}
    for g in ('G1','G7'):
        link=ref_root/g;target=reference[g]['path']
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        elif not link.is_symlink() or str(link.resolve())!=target:raise ValueError('reference link changed')
        baseline[g]=records(link)
        report['full'][g]=summarize_search(link,json.loads((link/'summary.json').read_text()))
    write_json(root/'summary.json',report)
    ctx=mp.get_context('spawn')
    try:
        result={}
        with ProcessPoolExecutor(args.workers,mp_context=ctx,initializer=init_phase1_worker,
                                 initargs=(ctx.Value('i',0),[int(g) for g in args.gpus.split(',')])) as pool:
            for name,config in search_variants().items():
                stats,result[name]=run_variant(pool,root,'full',name,config,scenes,inputs,protocol,manifest)
                report['full'][name]=summarize_search(root/'full'/name,stats)
                report['paired'][name+'-G7']=compare(baseline['G7'],result[name],'G7',name)
                if name=='G10':
                    report['paired']['G10-G8-96']=compare(result['G8-96'],result['G10'],'G8-96','G10')
                    report['paired']['G10-G9']=compare(result['G9'],result['G10'],'G9','G10')
                write_json(root/'summary.json',report)
            selection=choose_new_method(report['full'])
            selection.update(protocol_version=VERSION,heldout_scene_file=file_identity(holdout_file),
                             development_summaries={g:file_identity(root/'full'/g/'summary.json') for g in search_variants()})
            lock=root/'selection.json'
            if lock.exists() and json.loads(lock.read_text())!=selection:raise ValueError('locked validation method changed')
            write_json(lock,selection);report['selection']=selection
            write_json(root/'summary.json',report)
            from assignment.training.planning.search_report import build_report
            build_report(root)
            print(json.dumps(dict(validation_method_locked=selection['variant'])),flush=True)
            heldout_results={}
            configs={'G1':variants()['G1'],'G7':dict(variants()['G1'],candidates=48),selection['variant']:selection['config']}
            for name,config in configs.items():
                stats,heldout_results[name]=run_variant(pool,root,'heldout',name,config,heldout,inputs,protocol,manifest)
                report['heldout'][name]=summarize_search(root/'heldout'/name,stats)
                if name!='G1':report['heldout_paired'][name+'-G1']=compare(heldout_results['G1'],heldout_results[name],'G1',name)
                if name==selection['variant']:
                    report['heldout_paired'][name+'-G7']=compare(heldout_results['G7'],heldout_results[name],'G7',name)
                write_json(root/'summary.json',report)
        build_report(root)
        manifest['status']='complete'
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:
        manifest['ended_utc']=datetime.now(timezone.utc).isoformat();write_json(root/'run.json',manifest)


if __name__=='__main__':main()
