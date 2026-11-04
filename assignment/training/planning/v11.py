"""v1.1 bidirectional revision: G4 pilot -> locked G5 -> full G0-G5."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from assignment.training.planning.alignment import AlignmentConfig
from assignment.training.planning.diagnostics import aggregate, file_identity, write_json
from assignment.training.planning.evaluate import run_episode
from assignment.training.planning.phase1 import CONFIGS, init_phase1_worker, source_identity, validate_goals

PILOT_SCENES = [100038, 100069, 100078, 100152]
SEEDS = [0, 2, 3]
ALIGNMENT = asdict(AlignmentConfig())
BASE = dict(CONFIGS['B1'])
TASK = dict(target_count=4, horizon=1795, terminate_on_ground=False)
FORMULAS = dict(terminal='D[16]', tail_min='min(D[13:17])',
                tail_weighted='(D[13]+2*D[14]+3*D[15]+4*D[16])/10')
PROTOCOL_VERSION = 'v1.1-bidirectional-r2'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def variants(g5_scoring=None):
    pairs = dict(G0=('terminal', False), G1=('tail_min', False), G2=('tail_weighted', False),
                 G3=('terminal', True))
    result = {name: dict(BASE, scoring=score, realignment=align, realignment_config=ALIGNMENT)
              for name, (score, align) in pairs.items()}
    result['G4'] = dict(BASE, scoring='terminal', realignment=True, realignment_config=ALIGNMENT,
                        realignment_mode='bidirectional')
    if g5_scoring is not None:
        result['G5'] = dict(result['G4'], scoring=g5_scoring)
    return result


def choose_g5(g1, g2, pilot_n):
    """A one-box total difference across the 12 pilot episodes counts as basically tied."""
    difference = g1['mean_max_placed']-g2['mean_max_placed']
    choose_g1 = difference > 1/pilot_n+1e-12
    return dict(selected_source='G1' if choose_g1 else 'G2',
                scoring='tail_min' if choose_g1 else 'tail_weighted',
                rule='G1 only if mean peak exceeds G2 by more than 1/pilot_n; otherwise G2',
                mean_peak_difference_g1_minus_g2=difference, tie_tolerance_mean_peak=1/pilot_n)


def compatible_reuse(source, inputs, workers, gpus):
    """Explicit read-only links to validated G0-G3 records; old G4 is never imported."""
    source = Path(source).resolve()
    parent = json.loads((source/'run.json').read_text())
    p = parent['protocol']
    if p['workers'] != workers or p['gpus'] != gpus or p['task'] != TASK or p['sample_seeds'] != SEEDS:
        raise ValueError('reuse execution/task settings differ')
    if any(p['inputs'][k]['sha256'] != inputs[k]['sha256'] for k in inputs):
        raise ValueError('reuse input bytes differ')
    episodes = {}
    for stage in ('pilot', 'full'):
        for name in ('G0', 'G1', 'G2', 'G3'):
            if p['initial_variants'][name] != variants()[name]:
                raise ValueError(f'incompatible reused planner: {name}')
            group = source/stage/name
            if not (group/'run.json').exists():
                continue
            group_manifest = json.loads((group/'run.json').read_text())
            if group_manifest['config'] != variants()[name]:
                raise ValueError('reused group config differs; timeout variants are forbidden')
            for marker in sorted((group/'episodes').glob('*/complete.json')):
                if '.partial-' in marker.parent.name:
                    continue
                record = json.loads(marker.read_text())
                if record['protocol_id'] != group_manifest['protocol_id']:
                    raise ValueError('invalid reused completion marker')
                result = json.loads((marker.parent/'summary.json').read_text())
                scenes = PILOT_SCENES if stage=='pilot' else p['full_scenes']
                if result['scene_seed'] not in scenes or result['sample_seed'] not in SEEDS or result['variant']!=name:
                    raise ValueError('invalid reused case identity')
                if not (marker.parent/'video.mp4').exists():
                    raise ValueError('reused episode is missing video evidence')
                key = f'{stage}/{name}/{marker.parent.name}'
                episodes[key] = dict(source=str(marker.parent), protocol_id=record['protocol_id'],
                                     files={f: file_identity(marker.parent/f) for f in
                                            ('complete.json','summary.json','steps.jsonl','events.jsonl')})
    return dict(manifest=file_identity(source/'run.json'), episodes=episodes,
                execution_signature=parent['execution_signature'], runtime=parent['runtime'],
                rule='Only completed compatible G0-G3 are linked; old G4 excluded, never relabelled as G5')


def execute(job):
    result = run_episode(job)
    write_json(Path(job[10])/'complete.json', dict(protocol_id=job[11]['protocol_id'],
                                                 scene_seed=result['scene_seed'], sample_seed=result['sample_seed']))
    return result


def summarize_group(folder):
    results, times, alignment = [], [], dict(stagnant_blocks=0, checked_blocks=0, match_blocks=0,
                                            realignment_events=0, forward_realignments=0, backward_realignments=0,
                                            alignment_direction_reversals=0, closest_before_tail_blocks=0, decisions=0)
    for path in sorted((folder/'episodes').glob('*/complete.json')):
        episode = path.parent
        result = json.loads((episode/'summary.json').read_text(encoding='utf-8'))
        results.append(result)
        rows = [json.loads(s) for s in (episode/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
        events = [json.loads(s) for s in (episode/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        times.extend(r['decision_s'] for r in rows)
        alignment['decisions'] += len(rows)
        alignment['stagnant_blocks'] += sum(bool(r.get('stagnation_condition_met')) for r in rows)
        alignment['checked_blocks'] += sum(r.get('nearby_match_found',r.get('future_match_found')) is not None for r in rows)
        alignment['match_blocks'] += sum(bool(r.get('nearby_match_found',r.get('future_match_found'))) for r in rows)
        alignment['closest_before_tail_blocks'] += sum(r.get('selected_closest_block', 16)<13 for r in rows)
        jumps = [e for e in events if str(e.get('goal_change_reason', '')).startswith('realignment')]
        alignment['realignment_events'] += len(jumps)
        directions = [1 if e['active_goal_index']>e['previous_index'] else -1 for e in jumps]
        alignment['forward_realignments'] += directions.count(1)
        alignment['backward_realignments'] += directions.count(-1)
        alignment['alignment_direction_reversals'] += sum(a!=b for a,b in zip(directions,directions[1:]))
    summary = aggregate(results, times)
    summary['mechanism'] = alignment
    write_json(folder/'summary.json', summary)
    return summary, results


def paired_comparison(reference, treatment, bootstrap_draws=5000):
    base = {(r['scene_seed'], r['sample_seed']): r for r in reference}
    other = {(r['scene_seed'], r['sample_seed']): r for r in treatment}
    if base.keys() != other.keys():
        raise ValueError('paired comparison requires exactly the same scene/seed cases')
    delta = np.array([other[k]['max_placed']-base[k]['max_placed'] for k in sorted(base)], float)
    scenes = sorted({k[0] for k in base})
    scene_delta = np.array([np.mean([other[k]['max_placed']-base[k]['max_placed'] for k in base if k[0]==scene])
                            for scene in scenes])
    rng = np.random.default_rng(1103)
    boot = scene_delta[rng.integers(len(scenes), size=(bootstrap_draws, len(scenes)))].mean(-1)
    return dict(episodes=len(delta), scenes=len(scenes), mean_peak_delta=float(delta.mean()),
                improved=int((delta>0).sum()), tied=int((delta==0).sum()), regressed=int((delta<0).sum()),
                mean_peak_delta_ci95_scene_bootstrap=np.percentile(boot, [2.5, 97.5]).tolist(),
                bootstrap=dict(draws=bootstrap_draws, seed=1103, resampling_unit='whole scene with all sampler seeds'),
                cases=[dict(scene=k[0], sample_seed=k[1], peak_base=base[k]['max_placed'],
                            peak=other[k]['max_placed'], delta=other[k]['max_placed']-base[k]['max_placed'],
                            final_base=base[k]['final_count'], final=other[k]['final_count']) for k in sorted(base)])


def run_variant(pool, root, stage, name, config, scenes, inputs, protocol, manifest, single_check=False):
    folder = root/stage/name
    folder.mkdir(parents=True, exist_ok=True)
    group_protocol = digest(dict(protocol_id=manifest['protocol_id'], stage=stage, variant=name, config=config))
    write_json(folder/'run.json', dict(protocol_id=group_protocol, stage=stage, variant=name, config=config,
                                     scoring_formula=FORMULAS[config['scoring']], scene_seeds=scenes,
                                     sample_seeds=SEEDS, task=TASK, evidence=protocol['evidence']))
    pending = []
    reused = []
    for sample in SEEDS:
        for scene in scenes:
            episode = folder/'episodes'/f'scene-{scene}.sample-{sample}'
            marker = episode/'complete.json'
            imported = protocol.get('reuse', {}).get('episodes', {}).get(f'{stage}/{name}/{episode.name}')
            if imported is not None:
                if episode.exists():
                    if not episode.is_symlink() or str(episode.resolve()) != imported['source']:
                        raise ValueError('reuse link points to unexpected source')
                else:
                    episode.parent.mkdir(parents=True, exist_ok=True)
                    episode.symlink_to(imported['source'], target_is_directory=True)
                reused.append(imported)
                continue
            if marker.exists():
                previous = json.loads(marker.read_text(encoding='utf-8'))
                if previous['protocol_id'] != group_protocol:
                    raise ValueError(f'resume protocol mismatch: {episode}')
                continue
            if episode.exists():
                archived = episode.with_name(episode.name+'.partial-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
                if not episode.resolve().is_relative_to(root.resolve()) or not archived.resolve().is_relative_to(root.resolve()):
                    raise ValueError('partial episode archive escapes run directory')
                episode.rename(archived)
            job = ('goal', inputs['policy']['path'], inputs['world_model']['path'], config, scene, TASK,
                   128, str(episode/'video.mp4'), inputs['goals']['path'], sample, str(episode),
                   dict(label=name, tail_state_steps=56, keyframes=6, protocol_id=group_protocol,
                        protocol_version=protocol.get('version', PROTOCOL_VERSION)))
            pending.append(job)
    manifest.update(stage=stage, variant=name, planned_group_episodes=len(scenes)*len(SEEDS),
                    remaining_group_episodes=len(pending))
    write_json(root/'run.json', manifest)
    write_json(folder/'reuse.json', dict(episodes=reused, count=len(reused), method='read-only source symlinks'))
    def completed(result):
        runtime = result['runtime']
        # Physical device indices can differ; all numerical execution settings must match.
        signature = {k: v for k, v in runtime.items() if k not in ('cuda_visible_devices', 'device')}
        if manifest.get('execution_signature') not in (None, signature):
            raise ValueError('GPU/dtype/dependency mismatch inside the paired experiment')
        manifest['execution_signature'] = signature
        if runtime not in manifest['runtime']:
            manifest['runtime'].append(runtime)
        manifest['remaining_group_episodes'] -= 1
        manifest['completed_new_episodes'] += 1
        write_json(root/'run.json', manifest)
        print(json.dumps(dict(stage=stage, variant=name, scene=result['scene_seed'], sample=result['sample_seed'],
                              peak=result['max_placed'], final=result['final_count'],
                              remaining=manifest['remaining_group_episodes'])), flush=True)
    # First real G4 case is checked before launching the remaining eleven cases.
    if single_check and pending:
        first = pending.pop(0)
        result = pool.submit(execute, first).result()
        completed(result)
        checks = audit_bidirectional_episode(Path(first[10]))
        write_json(folder/'single-case-check.json', checks)
    futures = {pool.submit(execute, job): job for job in pending}
    for future in as_completed(futures):
        completed(future.result())
    summary, results = summarize_group(folder)
    if len(results) != len(scenes)*len(SEEDS):
        raise ValueError(f'incomplete group {stage}/{name}')
    return summary, results


def audit_bidirectional_episode(folder):
    rows = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    previous = 0
    counts = dict(forward=0, backward=0)
    for row in rows:
        if row['protocol_version'] != PROTOCOL_VERSION or row['alignment_search_mode']!='bidirectional':
            raise ValueError('G4 single-case check: incorrect protocol/mode')
        j = row['active_goal_index']
        if row.get('alignment_match_direction') is not None:
            if not row['stagnation_condition_met'] or row['matched_confirmations']!=3 or row['matched_goal_q']>=1:
                raise ValueError('G4 single-case check: unsupported alignment')
            if j != row['matched_goal_index']:
                raise ValueError('G4 single-case check: matched target was advanced in same call')
            counts[row['alignment_match_direction']] += 1
        elif j<previous:
            raise ValueError('G4 single-case check: backward move without real match')
        previous = j
    return dict(passed=True, episode=str(folder), decisions=len(rows), realignments=counts,
                scope='Scalar protocol/matching audit; manual video interpretation remains separate')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reuse-run', type=Path, help='Compatible previous v1.1 G0-G3 evidence; never import old G4')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    source = args.source_run.resolve()
    parent = json.loads((source/'run.json').read_text(encoding='utf-8'))
    scenes = parent['scene_seeds']
    if parent['status'] != 'complete' or len(scenes)!=20 or parent['sample_seeds'] != SEEDS:
        raise ValueError('source must be the complete fixed 20-scene/three-seed baseline')
    inputs = {k: file_identity(parent['inputs'][k]['path']) for k in ('policy', 'world_model', 'goals')}
    if any(inputs[k]['sha256'] != parent['inputs'][k]['sha256'] for k in inputs):
        raise ValueError('input checkpoint/goals bytes differ from original baseline')
    validate_goals(inputs['goals']['path'], scenes)
    if not set(PILOT_SCENES).issubset(scenes) or args.workers<1:
        raise ValueError('invalid pilot/worker settings')
    protocol = dict(version=PROTOCOL_VERSION, code=source_identity(), inputs=inputs, task=TASK,
                    design=file_identity(Path(__file__).resolve().parents[3]/'notes/v1.1/DESIGN.md'),
                    pilot_scenes=PILOT_SCENES, full_scenes=scenes, sample_seeds=SEEDS,
                    initial_variants=variants(), formulas=FORMULAS,
                    definitions=dict(G3='forward alignment + terminal', G4='bidirectional alignment + terminal',
                                     G5='bidirectional alignment + selected scoring', timeout_advance=False,
                                     bidirectional_same_call_official_advance=False),
                    pilot_selection=dict(primary='mean peak boxes', tie_tolerance=1/12,
                                         rule='G1 only if mean peak > G2 + 1/12; otherwise G2'),
                    full_baseline='same-runtime G0; compatible completed G0 may be reused; historical BF16 not pooled',
                    workers=args.workers, gpus=args.gpus,
                    evidence=dict(videos='all episodes', tail_state_steps=56, keyframes=6),
                    old_baseline=file_identity(source/'run.json'))
    if args.reuse_run:
        protocol['reuse'] = compatible_reuse(args.reuse_run, inputs, args.workers, args.gpus)
        prior = json.loads((args.reuse_run/'run.json').read_text())['protocol']
        if prior['full_scenes'] != scenes or prior['pilot_scenes'] != PILOT_SCENES:
            raise ValueError('reused scene sets differ from new fixed cohort')
    provenance = Path(__file__).resolve().parents[3]/'source-provenance.json'
    if provenance.exists():
        protocol['code']['snapshot_origin'] = json.loads(provenance.read_text(encoding='utf-8'))
    protocol_id = digest(protocol)
    root = args.out.resolve()
    if args.resume:
        manifest = json.loads((root/'run.json').read_text(encoding='utf-8'))
        if manifest['protocol_id'] != protocol_id:
            raise ValueError('resume requires identical source, inputs and precommitted settings')
        manifest.update(status='running', resumed_utc=datetime.now(timezone.utc).isoformat())
    else:
        root.mkdir(parents=True, exist_ok=False)
        manifest = dict(schema_version=1, status='running', protocol_id=protocol_id, protocol=protocol,
                        runtime=[], completed_new_episodes=0, started_utc=datetime.now(timezone.utc).isoformat())
        if args.reuse_run:
            manifest.update(execution_signature=protocol['reuse']['execution_signature'],
                            runtime=protocol['reuse']['runtime'],
                            reused_episodes=len(protocol['reuse']['episodes']))
    write_json(root/'run.json', manifest)
    report = dict(pilot={}, full={}, paired={}, contrasts={})
    ctx = mp.get_context('spawn')
    try:
        with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_phase1_worker,
                                 initargs=(ctx.Value('i', 0), [int(g) for g in args.gpus.split(',')])) as pool:
            for name, config in variants().items():
                stats, results = run_variant(pool, root, 'pilot', name, config, PILOT_SCENES,
                                             inputs, protocol, manifest, single_check=name=='G4')
                report['pilot'][name] = stats
                write_json(root/'summary.json', report)
                if name == 'G0':
                    fields = ('max_placed', 'final_count', 'first_placement_step', 'stage_steps', 'goal_index')
                    comparisons = []
                    for result in results:
                        old = json.loads((source/'B1/episodes'/f'scene-{result["scene_seed"]}.sample-{result["sample_seed"]}'/'summary.json').read_text())
                        comparisons.append(dict(scene=result['scene_seed'], sample_seed=result['sample_seed'],
                                                matching={key: old.get(key)==result.get(key) for key in fields}))
                    write_json(root/'g0-pilot-historical-check.json', dict(historical=file_identity(source/'run.json'),
                               original_runtime=parent['runtime'], current_runtime=manifest['runtime'], cases=comparisons,
                               policy='full fresh G0 is always run; historical outcomes not used as treatment control'))
            selection = choose_g5(report['pilot']['G1'], report['pilot']['G2'], 12)
            selection.update(protocol_version=PROTOCOL_VERSION, combined_group='G5',
                             config=variants(selection['scoring'])['G5'],
                             based_only_on='fixed pilot; full results are not used',
                             pilot_summaries={g: file_identity(root/'pilot'/g/'summary.json') for g in ('G1', 'G2', 'G4')})
            lock = root/'selection.json'
            if lock.exists() and json.loads(lock.read_text(encoding='utf-8')) != selection:
                raise ValueError('locked G5 choice differs from pilot; refusing to change it')
            write_json(lock, selection)
            print(json.dumps(dict(g5_locked=selection)), flush=True)
            report['pilot']['G5'], _ = run_variant(pool, root, 'pilot', 'G5', selection['config'],
                                                  PILOT_SCENES, inputs, protocol, manifest)
            write_json(root/'summary.json', report)
            from assignment.training.planning.v11_report import build_pilot_report
            build_pilot_report(root)
            reference = None
            full_results = {}
            for name, config in variants(selection['scoring']).items():
                stats, results = run_variant(pool, root, 'full', name, config, scenes, inputs, protocol, manifest)
                report['full'][name] = stats
                full_results[name] = results
                if name == 'G0':
                    reference = results
                else:
                    report['paired'][name] = paired_comparison(reference, results)
                for treatment, base in (('G4','G3'), ('G5','G4'), ('G5',selection['selected_source'])):
                    if treatment in full_results and base in full_results:
                        report['contrasts'][f'{treatment}-{base}'] = paired_comparison(full_results[base], full_results[treatment])
                write_json(root/'summary.json', report)
        from assignment.training.planning.v11_report import build_report
        build_report(root)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(root/'run.json', manifest)


if __name__ == '__main__':
    main()
