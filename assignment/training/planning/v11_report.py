"""Offline v1.1 evidence gallery; reads scalars and links existing videos only."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def episode_plot(folder, destination):
    import matplotlib.pyplot as plt
    rows = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    t = np.array([r['sim_time'] for r in rows])
    curves = np.array([r['selected_goal_distances_sq'] for r in rows], float)
    thresholds = np.array([r['goal_threshold_sq'] for r in rows], float)
    ratios = np.divide(curves, thresholds[:, None], out=np.full_like(curves, np.nan),
                       where=thresholds[:, None]>0)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    image = axes[0].pcolormesh(t, np.arange(1, 17)*.5, np.log10(np.maximum(ratios.T, 1e-4)),
                               shading='nearest', vmin=-2, vmax=2, cmap='coolwarm')
    fig.colorbar(image, ax=axes[0], label='log10(D / T)')
    axes[0].set_ylabel('Imagined time (s)')
    axes[0].axhline(6.25, color='black', linestyle='--', linewidth=1)
    axes[1].scatter(t, np.array([r['selected_closest_block'] for r in rows])*.5, s=4,
                    label='Closest across all 16 blocks')
    axes[1].scatter(t, np.array([r['selected_tail_closest_block'] for r in rows])*.5, s=4,
                    label='Closest within tail 4')
    axes[1].set_ylabel('Closest imagined time (s)')
    axes[1].legend(loc='upper right')
    axes[2].plot(t, [r['active_goal_index'] for r in rows], label='Active goal', drawstyle='steps-post')
    for key, color, name in [('stagnation_condition_met', 'orange', 'Stagnation gate'),
                              ('nearby_match_found', 'green', 'Qualified nearby match')]:
        mask = np.array([bool(r.get(key,r.get('future_match_found'))) for r in rows])
        axes[2].scatter(t[mask], np.array([r['active_goal_index'] for r in rows])[mask],
                        color=color, s=9, label=name)
    for row in rows:
        if str(row.get('goal_change_reason')).startswith('realignment'):
            for ax in axes:
                ax.axvline(row['sim_time'], color='purple', linewidth=1)
    axes[2].legend(loc='upper left')
    axes[2].set_xlabel('Real simulation time (s)')
    fig.suptitle(f'{folder.parents[1].name}: {folder.name}; distances are mechanism evidence only')
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def build_pilot_report(root):
    """Available before full evaluation; no full-cohort data affect G5 selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from assignment.training.planning.v11 import paired_comparison
    root = Path(root)
    summary = read(root/'summary.json')['pilot']
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    groups = list(summary)
    results = {g:[read(p.parent/'summary.json') for p in sorted((root/'pilot'/g/'episodes').glob('*/complete.json'))]
               for g in groups}
    paired = {g:paired_comparison(results['G0'], results[g]) for g in groups if g!='G0'}
    contrasts = {f'{t}-{b}':paired_comparison(results[b],results[t]) for t,b in [('G4','G3'),('G5','G4')]
                 if t in results and b in results}
    (out/'pilot-paired.json').write_text(json.dumps(dict(vs_g0=paired,contrasts=contrasts),indent=2))
    fig, axes = plt.subplots(1,3,figsize=(14,4.4),constrained_layout=True)
    x = np.arange(len(groups))
    axes[0].bar(x-.18,[summary[g]['mean_max_placed'] for g in groups],.36,label='Peak')
    axes[0].bar(x+.18,[summary[g]['mean_final_count'] for g in groups],.36,label='Final')
    axes[0].set(xticks=x,xticklabels=groups,ylabel='Mean boxes',ylim=(0,4))
    axes[0].legend()
    bottom = np.zeros(len(paired))
    for key,color in [('improved','#35a87c'),('tied','#b8bec8'),('regressed','#de7671')]:
        values = [paired[g][key] for g in paired]
        axes[1].bar(list(paired),values,bottom=bottom,label=key,color=color)
        bottom+=values
    axes[1].set(ylabel='Paired peak vs G0 (12 episodes)',ylim=(0,12))
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.27),ncol=3,fontsize=8)
    for offset,key,label in [(-.18,'forward_realignments','Forward'),(.18,'backward_realignments','Backward')]:
        axes[2].bar(x+offset,[summary[g]['mechanism'][key] for g in groups],.36,label=label)
    axes[2].set(xticks=x,xticklabels=groups,ylabel='Realignment events')
    axes[2].legend()
    fig.suptitle('PILOT ONLY: 4 scenes x 3 seeds; new G4=bidirectional, G5=combined')
    fig.savefig(out/'pilot-outcomes.png',dpi=150)
    plt.close(fig)
    gallery = []
    for label,pair in contrasts.items():
        t,b = label.split('-')
        for case in (min(pair['cases'],key=lambda c:c['delta']), max(pair['cases'],key=lambda c:c['delta'])):
            gallery.append(f'<h3>{label}: scene {case["scene"]}, seed {case["sample_seed"]}, peak delta {case["delta"]:+d}</h3><div class="pair">')
            for g in (b,t):
                rel=Path('pilot')/g/'episodes'/f'scene-{case["scene"]}.sample-{case["sample_seed"]}'
                stem=f'pilot-{g}-{case["scene"]}-{case["sample_seed"]}'
                episode_plot(root/rel,out/f'{stem}.png')
                url='../'+rel.as_posix()
                gallery.append(f'<div><h4>{g}</h4><video controls preload="none" src="{url}/video.mp4"></video>'
                               f'<p><a href="{url}/steps.jsonl">steps</a> | <a href="{url}/events.jsonl">events</a></p>'
                               f'<img src="{url}/timeline.png"><img src="{stem}.png"></div>')
            gallery.append('</div>')
    body='''<!doctype html><meta charset="utf-8"><title>New G4/G5 pilot</title>
<style>body{font:16px system-ui;margin:30px}img{max-width:100%}video{width:320px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}</style>
<h1>Bidirectional G4 / combined G5 pilot</h1><p>Four scenes only; this pilot overlaps full evaluation.
Old protocol G4 is not included. Examples show the largest paired gain/loss; behavior requires video review.</p>
<img src="pilot-outcomes.png"><p><a href="pilot-paired.json">All pilot paired cases</a> |
<a href="../selection.json">Locked G5 choice</a> | <a href="../summary.json">Statistics</a></p>'''+''.join(gallery)
    (out/'pilot.html').write_text(body,encoding='utf-8')


def build_report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    summary, selection = read(root/'summary.json'), read(root/'selection.json')
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    names = list(summary['full'])
    stats = summary['full']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    x = np.arange(len(names))
    axes[0].bar(x-.18, [stats[g]['mean_max_placed'] for g in names], .36, label='Peak')
    axes[0].bar(x+.18, [stats[g]['mean_final_count'] for g in names], .36, label='Final')
    axes[0].set(xticks=x, xticklabels=names, ylabel='Mean boxes', ylim=(0, 4))
    axes[0].legend()
    for k in (2, 3):
        axes[1].plot(names, [stats[g]['reached'][f'placed>={k}'] for g in names], marker='o', label=f'Peak >= {k}')
    axes[1].set(ylabel='Unconditional fraction', ylim=(0, 1))
    axes[1].legend()
    pairs = summary['paired']
    for i, (g, pair) in enumerate(pairs.items()):
        mean = pair['mean_peak_delta']
        lo, hi = pair['mean_peak_delta_ci95_scene_bootstrap']
        axes[2].errorbar(i, mean, yerr=np.array([[max(0, mean-lo)], [max(0, hi-mean)]]), fmt='o', capsize=5)
    axes[2].axhline(0, color='gray', linewidth=1)
    axes[2].set(xticks=np.arange(len(pairs)), xticklabels=list(pairs), ylabel='Paired mean peak delta vs fresh G0')
    axes[2].set_title('95% CI; resample whole scenes')
    fig.savefig(out/'outcomes.png', dpi=150)
    plt.close(fig)
    scenes = sorted({c['scene'] for p in pairs.values() for c in p['cases']})
    seeds = sorted({c['sample_seed'] for p in pairs.values() for c in p['cases']})
    fig, axes = plt.subplots(1, len(pairs), figsize=(15, 9), constrained_layout=True, squeeze=False)
    for ax, (g, pair) in zip(axes[0], pairs.items()):
        lookup = {(c['scene'], c['sample_seed']): c['delta'] for c in pair['cases']}
        data = np.array([[lookup[s, seed] for seed in seeds] for s in scenes])
        image = ax.imshow(data, vmin=-4, vmax=4, cmap='RdBu', aspect='auto')
        for i, j in np.ndindex(data.shape):
            ax.text(j, i, f'{int(data[i,j]):+d}', ha='center', va='center', fontsize=8)
        ax.set(xticks=np.arange(len(seeds)), xticklabels=seeds, yticks=np.arange(len(scenes)),
               yticklabels=scenes, title=f'{g} - G0', xlabel='Sampler seed')
    fig.colorbar(image, ax=list(axes[0]), label='Paired peak box delta')
    fig.savefig(out/'paired-cases.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, key, label in zip(axes, ('stagnant_blocks', 'match_blocks', 'realignment_events'),
                              ('Stagnant decisions', 'Decisions with nearby match', 'Confirmed realignments')):
        ax.bar(names, [stats[g]['mechanism'][key] for g in names])
        ax.set_title(label)
    fig.savefig(out/'alignment-mechanism.png', dpi=140)
    plt.close(fig)
    cases = [dict(variant=g, **case) for g, p in pairs.items() for case in p['cases']]
    with (out/'paired-cases.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    # Include censored observations rather than summarizing successful transitions alone.
    episode_rows = []
    for path in sorted((root/'full').glob('G*/episodes/*/complete.json')):
        r = read(path.parent/'summary.json')
        episode_rows.append({k: r.get(k) for k in ('variant', 'scene_seed', 'sample_seed', 'max_placed',
            'final_count', 'first_placement_step', 'second_placement_step', 'first_to_second_s',
            'second_reached', 'post_first_observed_s', 'decision_s_mean', 'decision_s_p95')})
    with (out/'episodes.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    gallery = []
    selected = []
    for g, pair in pairs.items():
        ordered = sorted(pair['cases'], key=lambda c: (c['delta'], c['scene'], c['sample_seed']))
        for label, pool in [('improved', [c for c in reversed(ordered) if c['delta']>0]),
                             ('tied', [c for c in ordered if c['delta']==0]),
                             ('regressed', [c for c in ordered if c['delta']<0])]:
            if not pool:
                continue
            case = pool[0]
            title = f'{g} {label}: scene {case["scene"]}, seed {case["sample_seed"]}, peak delta {case["delta"]:+d}'
            gallery.append(f'<h3>{html.escape(title)}</h3><div class="pair">')
            for variant in ('G0', g):
                rel = Path('full')/variant/'episodes'/f'scene-{case["scene"]}.sample-{case["sample_seed"]}'
                stem = f'{variant}-{case["scene"]}-{case["sample_seed"]}'
                episode_plot(root/rel, out/f'{stem}.png')
                url = '../'+rel.as_posix()
                gallery.append(f'<div><h4>{variant}</h4><video controls preload="none" src="{url}/video.mp4"></video>'
                               f'<p><a href="{url}/steps.jsonl">steps.jsonl</a> | <a href="{url}/events.jsonl">events</a></p>'
                               f'<img src="{url}/timeline.png"><img src="{stem}.png"></div>')
            gallery.append('</div>')
            selected.append(dict(variant=g, category=label, **case))
    (out/'representative-cases.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    table = ['<tr><th>Group</th><th>Mean peak</th><th>Mean final</th><th>Peak >=2</th><th>Peak >=3</th>'
             '<th>Second | first</th><th>Mean / P95 decision s</th><th>Improved / tied / regressed vs G0</th></tr>']
    for g in names:
        s = stats[g]
        counts = ' / '.join(str(pairs[g][k]) for k in ('improved', 'tied', 'regressed')) if g in pairs else '-'
        conditional = s['second_given_first']
        conditional_text = f'{conditional:.1%}' if conditional is not None else 'N/A'
        table.append(f'<tr><td>{g}</td><td>{s["mean_max_placed"]:.3f}</td><td>{s["mean_final_count"]:.3f}</td>'
                     f'<td>{s["reached"]["placed>=2"]:.1%}</td><td>{s["reached"]["placed>=3"]:.1%}</td>'
                     f'<td>{conditional_text}</td>')
        table[-1] += f'<td>{s["decision_s_mean"]:.3f} / {s["decision_s_p95"]:.3f}</td><td>{counts}</td></tr>'
    body = f'''<!doctype html><html><meta charset="utf-8"><title>v1.1 evidence</title>
<style>body{{font:16px system-ui;margin:30px;max-width:1500px}}img{{max-width:100%}}video{{width:360px;max-width:100%}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}</style>
<h1>v1.1 bidirectional G4 / combined G5 evidence</h1><p>G5 locked to {html.escape(selection['selected_source'])} scoring using only the fixed pilot.
G4 uses bidirectional alignment with terminal scoring; G5 combines bidirectional alignment and the selected score.
Full results include all 20 scenes × 3 seeds per group. G0 uses the same GPU/dtype; compatible prior records have explicit provenance.</p>
<p>Pilot overlaps full scenes and is not an independent test set. Differences in latent distances or goal index are mechanism evidence;
task outcomes determine benefit. Counts can drop when hand contact excludes a box.</p><table>{''.join(table)}</table>
<img src="outcomes.png"><img src="paired-cases.png"><img src="alignment-mechanism.png">
<p><a href="../summary.json">Statistics including G4-G3, G5-G4 and G5-selected-score contrasts</a> |
<a href="paired-cases.csv">All {len(cases)} paired cases</a> |
<a href="episodes.csv">All {len(episode_rows)} full episodes, including censored first-to-second observations</a> |
<a href="pilot.html">Pilot evidence</a></p>
<p>Examples are selected deterministically: largest gain, first tie, largest loss per treatment.
They have not been manually behavior-annotated. Purple lines indicate confirmed realignments; heatmap goals can change between decisions.</p>
{''.join(gallery)}</html>'''
    (out/'index.html').write_text(body, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    build_report(parser.parse_args().root)
