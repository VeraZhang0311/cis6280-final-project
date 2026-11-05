"""Aggregate complete planning runs, refusing incomplete scene/seed coverage."""
import argparse
import json
from pathlib import Path

from assignment.planning.experiments import ROOT, VARIANTS, validation_seeds
from assignment.planning.report import compare, read_reports, summarize_episodes


def summarize_experiment(root, stage, variants):
    offset, count, samples = {
        'smoke': (0, 2, [0]), 'screen': (0, 30, [0, 2, 3]),
        'confirm': (30, 270, [0, 2, 3]), 'full': (0, 300, [0, 2, 3]),
    }[stage]
    expected = {(scene, sample) for scene in validation_seeds()[offset:offset+count] for sample in samples}
    if 'baseline' not in variants:
        raise ValueError('baseline must be included')
    paths = {name: sorted((root/stage/name).glob('*.json')) for name in variants}
    reference = json.loads(read_reports(paths['baseline'])[1])
    rows = {}
    for name in variants:
        episodes, signature = read_reports(paths[name])
        metadata = json.loads(signature)
        if any(metadata[key] != reference[key] for key in ('policy', 'world_model', 'split', 'task')):
            raise ValueError(f'{name}: incompatible experiment metadata')
        if episodes.keys() != expected:
            raise ValueError(f'{name}: incomplete or unexpected scene/sample coverage')
        config = json.loads(paths[name][0].read_text())
        rows[name] = dict(**summarize_episodes(episodes), kind=config['kind'], config=config['config'])
        if name != 'baseline' and config['kind'] != 'df':
            comparison = compare(paths['baseline'], paths[name])
            rows[name].update(paired_delta=comparison['paired_delta'],
                              scene_bootstrap_95ci=comparison['scene_bootstrap_95ci'])
    equal_budget = [name for name in variants if rows[name]['kind'] != 'df'
                    and rows[name]['config'].get('candidates', 16) == 16]
    best = min(equal_budget, key=lambda name: (-rows[name]['mean_boxes'], rows[name]['ground_fall_rate'],
                                             rows[name]['agent_s_per_block']))
    return dict(stage=stage, scenes=count, samples=samples, episodes_per_variant=len(expected),
                variants=rows, best_equal_budget=best,
                selection_rule='Maximum mean boxes; ties use lower ground falls, then lower latency.',
                caveat='Development results require independent confirmation; smoke is a pipeline check.',
                input_reports={name: [str(path) for path in files] for name, files in paths.items()})


def markdown(report):
    lines = [f"# Planning {report['stage'].title()} Results", '',
             f"{report['scenes']} scenes; policy seeds {report['samples']}; "
             f"{report['episodes_per_variant']} episodes per variant.", '',
             '| Method | Mean boxes | >=2 | >=3 | 4 | Ground fall | ms/block | Delta vs baseline | 95% CI |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, row in report['variants'].items():
        ci = row.get('scene_bootstrap_95ci')
        interval = f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '-'
        delta = f"{row['paired_delta']:+.3f}" if 'paired_delta' in row else '-'
        lines.append(f"| {name} | {row['mean_boxes']:.3f} | {row['reached']['2']:.1%} | "
                     f"{row['reached']['3']:.1%} | {row['reached']['4']:.1%} | "
                     f"{row['ground_fall_rate']:.1%} | {1000*row['agent_s_per_block']:.1f} | {delta} | {interval} |")
    lines += ['', f"Best at 16 candidates: {report['best_equal_budget']}.", '', report['caveat'], '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'runs/planning')
    p.add_argument('--stage', choices=('smoke', 'screen', 'confirm', 'full'), default='screen')
    p.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    args = p.parse_args()
    report = summarize_experiment(args.root, args.stage, args.variants)
    (args.root/f'{args.stage}-summary.json').write_text(json.dumps(report, indent=2)+'\n')
    text = markdown(report)
    (args.root/f'{args.stage}-summary.md').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
