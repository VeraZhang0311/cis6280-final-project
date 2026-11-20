"""Audit completed HOLD traces without changing the frozen planner or evaluator."""
import argparse
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def audit(root, historical):
    manifest = read(root/'run.json')
    plan = manifest['experiment']['plan']
    expected = {(s, r) for s in plan['scene_seeds'] for r in plan['sample_seeds']}
    report = dict(groups={}, baseline_replay={},
                  note=('LOCAL-HOLD uses eligible for local safety. Generic lookahead counters '
                        'must not be interpreted as next-goal lookahead for this mode.'))
    baseline_traces = {}
    for mode in plan['groups']:
        paths = sorted((root/mode/'episodes').glob('*/episode.json'))
        episodes = [read(p) for p in paths]
        assert len(episodes) == len(expected)
        assert {(e['scene_seed'], e['sample_seed']) for e in episodes} == expected
        detail = []
        for path, episode in zip(paths, episodes):
            key = (episode['scene_seed'], episode['sample_seed'])
            assert episode['runtime']['gpu'] == manifest['runtime']['gpu']
            assert episode['runtime']['dtype'] == manifest['runtime']['dtype']
            traces = [json.loads(line) for line in (path.parent/'steps.jsonl').open()]
            counts = {r['control_step']: r['count'] for line in (path.parent/'counts.jsonl').open()
                      for r in [json.loads(line)]}
            held = [t for t in traces if t.get('hold_latched')]
            entries = [t for t in traces if t.get('hold_entered')]
            exits = [t for t in traces if t.get('hold_exited')]
            safe = [t for t in held if t['selection_rule'] == 'local_hold_safe_min_wrist']
            fallback = [t for t in held if t['selection_rule'] == 'local_hold_first4_mean_fallback']
            wrists = []
            for t in traces:
                assert (t['candidates'], t['generated_blocks'], t['imagined_blocks'],
                        t['execute_blocks']) == (32, 16, 16, 1)
            for i, t in enumerate(traces):
                close = (t['active_index'] == episode['goals']-1
                         and t['goal_distance'] < t['goal_threshold'])
                if t.get('hold_entered'):
                    prior = traces[i-1]
                    assert i > 0 and close and prior['active_index'] == episode['goals']-1
                    assert prior['goal_distance'] < prior['goal_threshold']
                if t.get('hold_exited'):
                    prior = traces[i-1]
                    assert not close and not prior['goal_distance'] < prior['goal_threshold']
            for t in safe:
                eligible = t['eligible']
                wrist, action = (t['candidate_first_block_wrist_rms'],
                                 t['candidate_first_block_action_rms'])
                chosen = min((i for i, ok in enumerate(eligible) if ok),
                             key=lambda i: (wrist[i], action[i], i))
                assert chosen == t['selected_candidate']
            for t in fallback:
                assert not any(t['eligible'])
                assert t['selected_candidate'] == min(range(32),
                    key=lambda i: (t['candidate_retention_distance'][i], i))
            for t in held:
                values = t.get('candidate_first_block_wrist_rms')
                if values is not None:
                    wrists.append(values[t['selected_candidate']])
            first_entry = entries[0]['control_step'] if entries else None
            if mode == 'A4-32':
                baseline_traces[key] = traces
            else:
                prefix = [t for t in traces if first_entry is None or t['control_step'] < first_entry]
                reference = baseline_traces[key]
                assert all(t['selected_candidate'] == reference[i]['selected_candidate']
                           and t['active_index'] == reference[i]['active_index']
                           for i, t in enumerate(prefix)), (mode, key, 'pre-HOLD differs')
            detail.append(dict(scene=key[0], sample_seed=key[1], peak=episode['peak_boxes'],
                               final=episode['final_boxes'], success=episode['success'],
                               decisions=episode['decisions'], latched_decisions=len(held),
                               entries=len(entries), exits=len(exits),
                               safe_candidate_decisions=len(safe), fallback_decisions=len(fallback),
                               selected_wrist_rms_sum=sum(wrists), wrist_samples=len(wrists),
                               entry_counts=[counts.get(t['control_step'], episode['start_count'])
                                             for t in entries]))
        total_held = sum(r['latched_decisions'] for r in detail)
        wrist_samples = sum(r['wrist_samples'] for r in detail)
        report['groups'][mode] = dict(
            episodes=len(detail), triggered_episodes=sum(r['entries'] > 0 for r in detail),
            failed_triggered_episodes=sum(r['entries'] > 0 and not r['success'] for r in detail),
            entries=sum(r['entries'] for r in detail), exits=sum(r['exits'] for r in detail),
            entries_below_four_boxes=sum(c < 4 for r in detail for c in r['entry_counts']),
            latched_decisions=total_held,
            safe_candidate_decisions=sum(r['safe_candidate_decisions'] for r in detail),
            fallback_decisions=sum(r['fallback_decisions'] for r in detail),
            safe_candidate_rate=(sum(r['safe_candidate_decisions'] for r in detail)/total_held
                                 if total_held and mode.endswith('LOCAL-HOLD') else None),
            selected_wrist_rms=(sum(r['selected_wrist_rms_sum'] for r in detail)/wrist_samples
                                if wrist_samples else None),
            mean_decisions=statistics.mean(r['decisions'] for r in detail), detail=detail)
        if mode == 'A4-32':
            old = {(r['scene_seed'], r['sample_seed']): r
                   for p in (historical/mode/'episodes').glob('*/episode.json') for r in [read(p)]}
            differences = [list(k) for e in episodes for k in [(e['scene_seed'], e['sample_seed'])]
                           if any(e[m] != old[k][m] for m in ('peak_boxes', 'final_boxes', 'success'))]
            report['baseline_replay'] = dict(matched=len(episodes)-len(differences),
                                             total=len(episodes), differences=differences)
    (root/'hold-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({mode: {k: v for k, v in row.items() if k != 'detail'}
                      for mode, row in report['groups'].items()}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--historical', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.historical)
