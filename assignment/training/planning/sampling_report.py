"""Offline G1/G6/G7 task, budget and timing evidence from saved scalars."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def episode(root,group,case):
    base=root/'reference/G1' if group=='G1' else root/'full'/group
    return base/'episodes'/f'scene-{case["scene"]}.sample-{case["sample_seed"]}'


def budget_plot(folder,destination):
    import matplotlib.pyplot as plt
    rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    times=[r['sim_time'] for r in rows]
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,constrained_layout=True)
    axes[0].step(times,[r.get('actual_candidates',16) for r in rows],where='post')
    axes[0].set(ylabel='Candidates',yticks=[16,32,48])
    ratios=[]
    gains=[]
    for r in rows:
        d=r.get('distance_now_sq',r['goal_distance_sq'])
        c=r.get('cumulative_best_costs',[min(r['candidate_scores'])])
        ratios.append((d-c[-1])/max(d,1e-8))
        gains.append((c[0]-c[-1])/max(d,1e-8))
    axes[1].plot(times,ratios,label='Final predicted relative improvement')
    axes[1].plot(times,gains,label='Gain from extra batches / D_now')
    axes[1].axhline(.20,color='gray',linestyle='--',label='G6 threshold 20%')
    axes[1].set_ylabel('Relative squared-distance change')
    axes[1].legend(loc='upper right')
    axes[2].plot(times,[r['decision_s'] for r in rows],label='Decision seconds')
    ax=axes[2].twinx()
    ax.step(times,[r.get('selected_batch',1) for r in rows],where='post',color='orange',alpha=.6)
    ax.set(ylabel='Selected batch',yticks=[1,2,3])
    axes[2].set(ylabel='Decision seconds',xlabel='Simulation seconds')
    fig.suptitle(f'{folder.parents[1].name}: {folder.name}; prediction gains are mechanism evidence only')
    fig.savefig(destination,dpi=130)
    plt.close(fig)


def build_report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from assignment.training.planning.v11_report import episode_plot
    root=Path(root)
    summary=read(root/'summary.json')
    stats=summary['full']
    groups=list(stats)
    out=root/'analysis'
    out.mkdir(exist_ok=True)
    x=np.arange(len(groups))
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    axes[0].bar(x-.18,[stats[g]['mean_max_placed'] for g in groups],.36,label='Peak')
    axes[0].bar(x+.18,[stats[g]['mean_final_count'] for g in groups],.36,label='Final')
    axes[0].set(xticks=x,xticklabels=groups,ylabel='Mean boxes',ylim=(0,4))
    axes[0].legend()
    axes[1].bar(groups,[stats[g]['sampling']['mean_candidates'] for g in groups])
    axes[1].set(ylabel='Mean candidates per decision',ylim=(0,50))
    axes[2].bar(x-.18,[stats[g]['decision_s_mean'] for g in groups],.36,label='Mean')
    axes[2].bar(x+.18,[stats[g]['decision_s_p95'] for g in groups],.36,label='P95')
    axes[2].set(xticks=x,xticklabels=groups,ylabel='Decision seconds')
    axes[2].legend()
    fig.suptitle('G1 reference / G6 adaptive 16-48 / G7 one batch of 48; no pilot')
    fig.savefig(out/'task-and-cost.png',dpi=150)
    plt.close(fig)
    pairs=summary['paired']
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for g in groups:
        counts=stats[g]['sampling']['candidate_count_distribution']
        total=stats[g]['sampling']['decisions']
        axes[0].plot([16,32,48],[counts[str(k)]/total for k in (16,32,48)],marker='o',label=g)
    axes[0].set(xticks=[16,32,48],xlabel='Actual candidates',ylabel='Fraction of decisions',ylim=(0,1))
    axes[0].legend()
    for i,(name,pair) in enumerate(pairs.items()):
        value=pair['mean_peak_delta'];lo,hi=pair['mean_peak_delta_ci95_scene_bootstrap']
        axes[1].errorbar(i,value,yerr=np.array([[max(0,value-lo)],[max(0,hi-value)]]),fmt='o',capsize=5)
    axes[1].axhline(0,color='gray',linewidth=1)
    axes[1].set(xticks=np.arange(len(pairs)),xticklabels=list(pairs),ylabel='Paired mean peak difference',
                title='95% CI; whole-scene bootstrap')
    fig.savefig(out/'budgets-and-pairs.png',dpi=150)
    plt.close(fig)
    paired_rows=[dict(comparison=name,**c) for name,pair in pairs.items() for c in pair['cases']]
    with (out/'paired-cases.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired_rows[0]));writer.writeheader();writer.writerows(paired_rows)
    episode_rows=[]
    for g in groups:
        base=root/'reference/G1' if g=='G1' else root/'full'/g
        for marker in sorted((base/'episodes').glob('*/complete.json')):
            r=read(marker.parent/'summary.json')
            episode_rows.append({k:r.get(k) for k in ('variant','scene_seed','sample_seed','max_placed','final_count',
                'first_placement_step','second_placement_step','first_to_second_s','second_reached','post_first_observed_s',
                'decision_s_mean','decision_s_p95')})
    with (out/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(episode_rows[0]));writer.writeheader();writer.writerows(episode_rows)
    table=['<tr><th>Group</th><th>Peak / final</th><th>Peak ≥2 / ≥3</th><th>Mean candidates</th>'
           '<th>Mean / P95 seconds</th><th>Second / third batch rate</th><th>Mean policy calls</th></tr>']
    for g,s in stats.items():
        b=s['sampling']
        table.append(f'<tr><td>{g}</td><td>{s["mean_max_placed"]:.3f} / {s["mean_final_count"]:.3f}</td>'
            f'<td>{s["reached"]["placed>=2"]:.1%} / {s["reached"]["placed>=3"]:.1%}</td>'
            f'<td>{b["mean_candidates"]:.2f}</td><td>{s["decision_s_mean"]:.3f} / {s["decision_s_p95"]:.3f}</td>'
            f'<td>{b["second_batch_rate"]:.1%} / {b["third_batch_rate"]:.1%}</td><td>{b["policy_sample_calls_mean"]:.2f}</td></tr>')
    gallery=[];selected=[]
    for label,pair in pairs.items():
        treatment,base=label.split('-')
        ordered=sorted(pair['cases'],key=lambda c:(c['delta'],c['scene'],c['sample_seed']))
        for category,pool in [('improved',[c for c in reversed(ordered) if c['delta']>0]),
                              ('tied',[c for c in ordered if c['delta']==0]),('regressed',[c for c in ordered if c['delta']<0])]:
            if not pool:continue
            case=pool[0]
            gallery.append(f'<h3>{label} {category}: scene {case["scene"]}, seed {case["sample_seed"]}, delta {case["delta"]:+d}</h3><div class="pair">')
            for group in (base,treatment):
                folder=episode(root,group,case)
                stem=f'{group}-{case["scene"]}-{case["sample_seed"]}'
                episode_plot(folder,out/f'{stem}-latents.png')
                budget_plot(folder,out/f'{stem}-budget.png')
                url='../'+folder.relative_to(root).as_posix()
                gallery.append(f'<div><h4>{group}</h4><video controls preload="none" src="{url}/video.mp4"></video>'
                    f'<p><a href="{url}/steps.jsonl">steps.jsonl</a> | <a href="{url}/events.jsonl">events</a></p>'
                    f'<img src="{url}/timeline.png"><img src="{stem}-budget.png"><img src="{stem}-latents.png"></div>')
            gallery.append('</div>');selected.append(dict(comparison=label,category=category,**case))
    (out/'representative-cases.json').write_text(json.dumps(selected,indent=2))
    body='''<!doctype html><meta charset="utf-8"><title>G6/G7 sampling comparison</title>
<style>body{font:16px system-ui;margin:30px;max-width:1500px}img{max-width:100%}video{width:340px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}td,th{padding:8px;border-bottom:1px solid #ddd}</style>
<h1>G1 / G6 / G7 full development comparison</h1>
<p>20 fixed scenes × seeds 0,2,3 per group. No pilot. G1 is the fixed-16 baseline described below.
G6 and G7 use its official goal switching and tail-min score, with no realignment.</p>
<p>G6 samples batches of 16, up to 48. G7 samples 48 once; equal maximum candidate budgets do not imply equal compute.
Batch shape can change RNG consumption/numerical results; candidate prefixes are not assumed identical.</p>
<p>G1 was selected from earlier development results on these same scenes. This is not an independent held-out test.
Task outcomes determine benefit; predicted distance gains are mechanism evidence only.</p>'''
    body+='<p>Baseline: '+html.escape(summary.get('baseline_mode','Explicitly referenced fixed-16 G1'))+'.</p>'
    body+='<table>'+''.join(table)+'</table><img src="task-and-cost.png"><img src="budgets-and-pairs.png">'
    body+='<p><a href="../summary.json">All statistics, stop reasons and batch gains</a> | <a href="../run.json">Protocol and reference provenance</a> | <a href="paired-cases.csv">Paired cases</a> | <a href="episodes.csv">All episode outcomes including censored transitions</a></p>'
    body+='<p>Examples are deterministic largest gain / first tie / largest loss; behavior has not been manually annotated.</p>'+''.join(gallery)
    (out/'index.html').write_text(body,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    build_report(p.parse_args().root)
