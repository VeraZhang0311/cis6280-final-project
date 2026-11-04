"""Direct full-cohort G6/G7 sampling comparison against the frozen completed G1."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
from collections import Counter

import numpy as np

from assignment.training.planning.diagnostics import file_identity, write_json
from assignment.training.planning.phase1 import init_phase1_worker, source_identity, validate_goals
from assignment.training.planning.v11 import TASK, SEEDS, variants, digest, paired_comparison, run_variant

VERSION='v1.1-sampling-r3'


def sampling_variants():
    base=variants()['G1']
    return dict(G6=dict(base, adaptive_sampling=True, max_candidates=48, min_improvement=.20),
                G7=dict(base, candidates=48))


def summarize_sampling(folder, task_summary):
    rows=[]
    for marker in sorted((Path(folder)/'episodes').glob('*/complete.json')):
        rows.extend(json.loads(s) for s in (marker.parent/'steps.jsonl').read_text().splitlines())
    counts=Counter(int(r.get('actual_candidates',16)) for r in rows)
    calls=[r.get('policy_sample_calls',1) for r in rows]
    gains=[b['extra_batch_cost_gain'] for r in rows for b in r.get('sampling_batches',[])
           if b.get('extra_batch_cost_gain') is not None]
    relative_gains=[(r['cumulative_best_costs'][0]-r['cumulative_best_costs'][-1])/r['distance_now_sq']
                    for r in rows if len(r.get('cumulative_best_costs',[]))>1 and r.get('distance_now_sq',0)>1e-8]
    result=dict(task_summary)
    result['sampling']=dict(decisions=len(rows), mean_candidates=float(np.mean([r.get('actual_candidates',16) for r in rows])),
        candidate_count_distribution={str(k):counts.get(k,0) for k in (16,32,48)},
        second_batch_rate=float(np.mean([n>=2 for n in calls])),
        third_batch_rate=float(np.mean([n>=3 for n in calls])),
        policy_sample_calls_total=int(sum(calls)), policy_sample_calls_mean=float(np.mean(calls)),
        wm_rollout_calls_total=int(sum(r.get('wm_rollout_calls',1) for r in rows)),
        selected_batch_distribution=dict(Counter(str(r.get('selected_batch',1)) for r in rows)),
        stop_reasons=dict(Counter(r.get('stop_reason','fixed_budget') for r in rows)),
        extra_batches=len(gains), mean_extra_batch_cost_gain=float(np.mean(gains)) if gains else None,
        mean_relative_extra_batch_gain=float(np.mean(relative_gains)) if relative_gains else None,
        metadata_source='recorded; missing G1 fixed-16 counts inferred only from validated frozen config')
    return result


def reference_metadata(source, workers, gpus, require_same_execution=True):
    source=Path(source).resolve()
    manifest=json.loads((source/'run.json').read_text())
    p=manifest['protocol']
    if manifest['status']!='complete' or p['task']!=TASK or p['sample_seeds']!=SEEDS:
        raise ValueError('reference must be complete and have identical execution/task settings')
    if require_same_execution and (p['workers']!=workers or p['gpus']!=gpus):
        raise ValueError('reuse requires identical worker/GPU allocation; use a fresh G1 instead')
    group=source/'full/G1'
    group_run=json.loads((group/'run.json').read_text())
    if group_run['config']!=variants()['G1']:
        raise ValueError('reference is not the frozen fixed-16 G1')
    expected={(s,k) for s in p['full_scenes'] for k in SEEDS}
    cases=[]
    for marker in sorted((group/'episodes').glob('*/complete.json')):
        result=json.loads((marker.parent/'summary.json').read_text())
        if json.loads(marker.read_text())['protocol_id']!=group_run['protocol_id'] or result['variant']!='G1':
            raise ValueError('reference case protocol mismatch')
        cases.append((result['scene_seed'],result['sample_seed']))
    if len(expected)!=60 or len(cases)!=60 or set(cases)!=expected:
        raise ValueError('reference must contain exactly the same 20 scenes x three seeds')
    metadata=dict(path=str(group), manifest=file_identity(source/'run.json'), protocol_id=manifest['protocol_id'],
                  group_manifest=file_identity(group/'run.json'), summary=file_identity(group/'summary.json'),
                  cases=[{f:file_identity(marker.parent/f) for f in ('complete.json','summary.json','steps.jsonl','events.jsonl')}
                         for marker in sorted((group/'episodes').glob('*/complete.json'))],
                  note='Historical G1 remains unchanged; comparison is development reuse, not independent test data')
    return manifest,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--fresh-reference',action='store_true',help='Rerun full G1 in the new allocation; do not pool historical outcomes')
    args=p.parse_args()
    parent,reference=reference_metadata(args.reference_run,args.workers,args.gpus,not args.fresh_reference)
    if args.fresh_reference:
        reference['note']='Source cohort/input provenance only; historical G1 outcomes not pooled. Fresh full G1 supplies the control.'
    old=parent['protocol']
    scenes=old['full_scenes']
    inputs={k:file_identity(v['path']) for k,v in old['inputs'].items()}
    if any(inputs[k]['sha256']!=v['sha256'] for k,v in old['inputs'].items()):
        raise ValueError('frozen checkpoint/goals bytes changed')
    validate_goals(inputs['goals']['path'],scenes)
    repo=Path(__file__).resolve().parents[3]
    code=source_identity()
    if (repo/'source-provenance.json').exists():
        code['snapshot_origin']=json.loads((repo/'source-provenance.json').read_text())
    protocol=dict(version=VERSION,base_variant='G1',scoring='tail_min',goal_selection='official',
                  code=code,design=file_identity(repo/'notes/v1.1/DESIGN.md'),inputs=inputs,task=TASK,
                  full_scenes=scenes,sample_seeds=SEEDS,pilot=False,workers=args.workers,gpus=args.gpus,
                  variants=dict(G1=variants()['G1'],**sampling_variants()) if args.fresh_reference else sampling_variants(),
                  reference=reference,fresh_reference=args.fresh_reference,
                  sampling=dict(G6=dict(mode='adaptive',batch_size=16,max_candidates=48,min_improvement=.20),
                                G7=dict(mode='fixed',batch_size=48,max_candidates=48)),
                  stop_rule='after 16/32: stop if D_now<=1e-8 or C_best<T or r>=.20; at 48 stop unconditionally',
                  ties='earliest global candidate',evidence=dict(videos='all episodes',tail_state_steps=56,keyframes=6),
                  comparison='Same scene/seed task pairing; G6 and G7 candidate prefixes need not be numerically identical')
    protocol_id=digest(protocol)
    root=args.out.resolve()
    if args.resume:
        manifest=json.loads((root/'run.json').read_text())
        if manifest['protocol_id']!=protocol_id:
            raise ValueError('resume requires identical source, reference and settings')
        manifest.update(status='running',resumed_utc=datetime.now(timezone.utc).isoformat())
    else:
        root.mkdir(parents=True,exist_ok=False)
        manifest=dict(schema_version=1,status='running',protocol_id=protocol_id,protocol=protocol,
                      started_utc=datetime.now(timezone.utc).isoformat(),
                      runtime=[] if args.fresh_reference else parent['runtime'],
                      completed_new_episodes=0,reference_episodes=0 if args.fresh_reference else 60)
        if not args.fresh_reference:manifest['execution_signature']=parent['execution_signature']
    write_json(root/'run.json',manifest)
    link=root/'reference/G1'
    link.parent.mkdir(exist_ok=True)
    target=str(root/'full/G1') if args.fresh_reference else reference['path']
    if not link.is_symlink() and not link.exists():link.symlink_to(target,target_is_directory=True)
    elif not link.is_symlink() or str(link.resolve())!=target:
        raise ValueError('reference link does not match frozen G1 source')
    report=dict(protocol_version=VERSION,pilot=False,reference=reference,full={},paired={},
                baseline_mode='Fresh G1 on the same allocation' if args.fresh_reference else 'Reused completed G1 with identical allocation settings')
    if not args.fresh_reference:
        baseline=[json.loads((m.parent/'summary.json').read_text()) for m in sorted((link/'episodes').glob('*/complete.json'))]
        report['full']['G1']=summarize_sampling(link,json.loads((link/'summary.json').read_text()))
    write_json(root/'summary.json',report)
    ctx=mp.get_context('spawn')
    try:
        results={}
        with ProcessPoolExecutor(args.workers,mp_context=ctx,initializer=init_phase1_worker,
                                 initargs=(ctx.Value('i',0),[int(g) for g in args.gpus.split(',')])) as pool:
            if args.fresh_reference:
                stats,baseline=run_variant(pool,root,'full','G1',variants()['G1'],scenes,inputs,protocol,manifest)
                report['full']['G1']=summarize_sampling(root/'full/G1',stats)
                write_json(root/'summary.json',report)
            for name,config in sampling_variants().items():
                stats,records=run_variant(pool,root,'full',name,config,scenes,inputs,protocol,manifest)
                results[name]=records
                report['full'][name]=summarize_sampling(root/'full'/name,stats)
                report['paired'][name+'-G1']=paired_comparison(baseline,records)
                if name=='G7':report['paired']['G6-G7']=paired_comparison(records,results['G6'])
                write_json(root/'summary.json',report)
        from assignment.training.planning.sampling_report import build_report
        build_report(root)
        manifest['status']='complete'
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc))
        raise
    finally:
        manifest['ended_utc']=datetime.now(timezone.utc).isoformat()
        write_json(root/'run.json',manifest)


if __name__=='__main__':main()
