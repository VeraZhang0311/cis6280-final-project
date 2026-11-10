"""Validate complete integration evidence and render its paired comparison table."""
import argparse
import json
from pathlib import Path

from assignment.training.planning.evaluate_goal_actor import paired_comparisons, summarize


def report(root):
    manifest = json.loads((root/'run.json').read_text())
    plan = manifest['experiment']['plan']
    expected = {(s, r) for s in plan['scene_seeds'] for r in plan['sample_seeds']}
    results = {}
    for mode in plan['groups']:
        paths = sorted((root/mode/'episodes').glob('*/episode.json'))
        rows = [json.loads(path.read_text()) for path in paths]
        if len(rows) != len(expected) or {(r['scene_seed'], r['sample_seed']) for r in rows} != expected:
            raise ValueError(f'{mode}: missing, duplicated, or unexpected episodes')
        for path, row in zip(paths, rows):
            if row['mode'] != mode or row['runtime']['gpu'] != manifest['runtime']['gpu']:
                raise ValueError(f'{mode}: incompatible episode identity/hardware')
            for key in ('device', 'dtype'):
                if row['runtime'][key] != manifest['runtime'][key]:
                    raise ValueError(f'{mode}: incompatible {key}')
            for name in ('video.mp4', 'steps.jsonl', 'counts.jsonl'):
                if not (path.parent/name).is_file() or not (path.parent/name).stat().st_size:
                    raise ValueError(f'{mode}: missing {name}')
        results[mode] = rows
    comparison = dict(groups={mode: summarize(rows) for mode, rows in results.items()},
                      paired_comparisons=paired_comparisons(results),
                      episodes=sum(map(len, results.values())), runtime=manifest['runtime'],
                      experiment=manifest['experiment'], code=manifest['code'])
    if comparison['episodes'] != plan['expected_episodes']:
        raise ValueError('Unexpected experiment total')
    lines = ['# v2.0 integration evaluation', '',
             f"{len(expected)} episodes/group, {comparison['episodes']} total; "
             f"GPU: {manifest['runtime']['gpu']}; dtype: {manifest['runtime']['dtype']}.", '',
             '| Group | Peak boxes | Final boxes | Success | Four-box retention | Mean / P95 ms |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode, group in comparison['groups'].items():
        retention = group['target_retention']
        rate = f"{retention['retained']}/{retention['at_risk']}" if retention['at_risk'] else 'n/a'
        successes = sum(row['success'] for row in results[mode])
        lines.append(f"| {mode} | {group['mean_peak_boxes']:.3f} | {group['mean_final_boxes']:.3f} | "
                     f"{successes}/{len(expected)} ({group['success']:.1%}) | {rate} | "
                     f"{1000*group['mean_decision_s']:.1f} / {1000*group['p95_decision_s']:.1f} |")
    lines += ['', 'Paired differences against the fresh A4-32 baseline, with scene-bootstrap 95% intervals:', '',
              '| Group | Peak delta [95% CI] | Final delta [95% CI] | Success delta [95% CI], pp |',
              '| --- | ---: | ---: | ---: |']
    for mode in plan['groups']:
        key = f'{mode}_minus_A4-32'
        if key not in comparison['paired_comparisons']:
            continue
        metrics = comparison['paired_comparisons'][key]['metrics']
        cells = []
        for metric in ('peak_boxes', 'final_boxes', 'success'):
            value = metrics[metric]
            scale = 100 if metric == 'success' else 1
            low, high = value['ci95']
            cells.append(f"{scale*value['difference']:+.3f} [{scale*low:+.3f}, {scale*high:+.3f}]")
        lines.append(f"| {mode} | {' | '.join(cells)} |")
    lines += ['', 'These are paired development results with multiple exploratory comparisons. '
              'Do not treat a selected best variant as independently confirmed. '
              'Peak four boxes and official success are separate metrics.', '']
    return comparison, '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    comparison, markdown = report(args.root)
    (args.root/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    (args.root/'RESULTS.md').write_text(markdown)
    print(markdown)


if __name__ == '__main__':
    main()
