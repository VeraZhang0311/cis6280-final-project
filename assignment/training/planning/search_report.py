"""Offline development/heldout evidence for G8-32/G8-96/G9/G10."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def model_plot(folder,destination):
    import matplotlib.pyplot as plt
    rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    times=[r['sim_time'] for r in rows]
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,constrained_layout=True)
    axes[0].plot(times,[r.get('trajectory_evaluations',r.get('actual_candidates',16)) for r in rows],label='Full trajectory evaluations')
    axes[0].plot(times,[r.get('valid_cached_candidates',0) for r in rows],label='Reconditioned cached plans')
    axes[0].legend();axes[0].set_ylabel('Plans')
    axes[1].plot(times,[r.get('policy_sample_calls',1) for r in rows],label='Policy calls')
    axes[1].plot(times,[r.get('wm_rollout_calls',1) for r in rows],label='WM calls')
    axes[1].legend();axes[1].set_ylabel('Actual batches')
    gains=[]
    for row in rows:
        costs=row.get('cumulative_best_costs',[min(row['candidate_scores'])])
        gains.append((costs[0]-costs[-1])/max(row['goal_distance_sq'],1e-8))
    axes[2].plot(times,gains,label='Search gain / current real distance')
    axes[2].set(ylabel='Within-decision predicted gain',xlabel='Simulation seconds')
    twin=axes[2].twinx();twin.plot(times,[r['decision_s'] for r in rows],color='orange',alpha=.4)
    twin.set_ylabel('Decision seconds')
    fig.suptitle(folder.name+'; costs are mechanism evidence only')
    fig.savefig(destination,dpi=130);plt.close(fig)


def outcome_plot(stats,pairs,path,title):
    import matplotlib.pyplot as plt
    names=list(stats);x=np.arange(len(names))
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    axes[0].bar(x-.18,[stats[g]['mean_max_placed'] for g in names],.36,label='Peak')
    axes[0].bar(x+.18,[stats[g]['mean_final_count'] for g in names],.36,label='Final')
    axes[0].set(xticks=x,xticklabels=names,ylabel='Mean boxes',ylim=(0,4));axes[0].legend()
    axes[1].bar(x-.18,[stats[g]['decision_s_mean'] for g in names],.36,label='Mean')
    axes[1].bar(x+.18,[stats[g]['decision_s_p95'] for g in names],.36,label='P95')
    axes[1].set(xticks=x,xticklabels=names,ylabel='Decision seconds');axes[1].legend()
    for i,(name,pair) in enumerate(pairs.items()):
        m=pair['mean_peak_delta'];lo,hi=pair['mean_peak_delta_ci95_scene_bootstrap']
        axes[2].errorbar(i,m,yerr=np.array([[max(0,m-lo)],[max(0,hi-m)]]),fmt='o',capsize=4)
    axes[2].axhline(0,color='gray',lw=1)
    axes[2].set(xticks=np.arange(len(pairs)),xticklabels=list(pairs),ylabel='Paired mean peak delta')
    axes[2].tick_params(axis='x',rotation=40)
    axes[2].set_title('95% whole-scene bootstrap CI')
    fig.suptitle(title);fig.savefig(path,dpi=140);plt.close(fig)


def table(stats):
    rows=['<tr><th>Group</th><th>Peak / final</th><th>Peak >=2 / >=3</th><th>Mean / P95 ms</th>'
          '<th>Trajectory / latent predictions per decision</th><th>Policy / WM calls</th><th>Policy-generated blocks</th></tr>']
    for g,s in stats.items():
        c=s['search']
        rows.append(f'<tr><td>{g}</td><td>{s["mean_max_placed"]:.3f} / {s["mean_final_count"]:.3f}</td>'
            f'<td>{s["reached"]["placed>=2"]:.1%} / {s["reached"]["placed>=3"]:.1%}</td>'
            f'<td>{1000*s["decision_s_mean"]:.1f} / {1000*s["decision_s_p95"]:.1f}</td>'
            f'<td>{c["mean_trajectory_evaluations"]:.1f} / {c["mean_future_latent_predictions"]:.1f}</td>'
            f'<td>{c["mean_policy_calls"]:.2f} / {c["mean_wm_calls"]:.2f}</td>'
            f'<td>{c["mean_policy_action_blocks"]:.1f}</td></tr>')
    return '<table>'+''.join(rows)+'</table>'


def build_report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from assignment.training.planning.v11_report import episode_plot
    root=Path(root);summary=read(root/'summary.json')
    out=root/'analysis';out.mkdir(exist_ok=True)
    outcome_plot(summary['full'],summary['paired'],out/'development.png','Development: 20 scenes x 3 seeds; G1/G7 explicit frozen references')
    stats=summary['full'];curve=['G1','G8-32','G7','G8-96']
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    budgets=[16,32,48,96]
    axes[0].plot(budgets,[stats[g]['mean_max_placed'] for g in curve],marker='o',label='Peak')
    axes[0].plot(budgets,[stats[g]['mean_final_count'] for g in curve],marker='o',label='Final')
    axes[0].set(xlabel='Fixed policy candidates',ylabel='Mean boxes');axes[0].legend()
    axes[1].scatter([s['decision_s_mean'] for s in stats.values()],[s['mean_max_placed'] for s in stats.values()])
    for g,s in stats.items():axes[1].annotate(g,(s['decision_s_mean'],s['mean_max_placed']))
    axes[1].set(xlabel='Mean decision seconds',ylabel='Mean peak boxes')
    fig.savefig(out/'budget-and-time.png',dpi=140);plt.close(fig)
    if summary['heldout']:
        outcome_plot(summary['heldout'],summary['heldout_paired'],out/'heldout.png','New fixed scenes: separate validation; method locked before outcomes')
    pairrows=[];episode_rows=[];gallery=[];selected=[]
    for stage,key,pairs_key in [('development','full','paired'),('heldout','heldout','heldout_paired')]:
        pairs=summary[pairs_key]
        for label,pair in pairs.items():
            pairrows.extend(dict(stage=stage,comparison=label,**c) for c in pair['cases'])
        for g in summary[key]:
            folder=(root/'reference'/g) if stage=='development' and g in ('G1','G7') else root/key/g
            for marker in sorted((folder/'episodes').glob('*/complete.json')):
                r=read(marker.parent/'summary.json')
                episode_rows.append(dict(stage=stage,**{k:r.get(k) for k in ('variant','scene_seed','sample_seed','max_placed',
                    'final_count','first_placement_step','second_placement_step','first_to_second_s','second_reached',
                    'post_first_observed_s','decision_s_mean','decision_s_p95')}))
        # A few largest gains/losses; full evidence remains accessible for every case.
        for label,pair in pairs.items():
            ordered=sorted(pair['cases'],key=lambda c:(c['delta'],c['scene'],c['sample_seed']))
            for category,candidates in [('improved',[c for c in reversed(ordered) if c['delta']>0]),
                                         ('regressed',[c for c in ordered if c['delta']<0])]:
                if not candidates:continue
                case=candidates[0]
                gallery.append(f'<h3>{stage} {label} {category}: {case["scene"]}, seed {case["sample_seed"]}, delta {case["delta"]:+d}</h3><div class="pair">')
                for g in (pair['baseline'],pair['treatment']):
                    folder=(root/'reference'/g) if stage=='development' and g in ('G1','G7') else root/key/g
                    folder=folder/'episodes'/f'scene-{case["scene"]}.sample-{case["sample_seed"]}'
                    stem=f'{stage}-{g}-{case["scene"]}-{case["sample_seed"]}'
                    episode_plot(folder,out/f'{stem}-latents.png');model_plot(folder,out/f'{stem}-search.png')
                    url='../'+folder.relative_to(root).as_posix()
                    gallery.append(f'<div><h4>{g}</h4><video controls preload="none" src="{url}/video.mp4"></video>'
                        f'<p><a href="{url}/steps.jsonl">steps</a> | <a href="{url}/events.jsonl">events</a></p>'
                        f'<img src="{url}/timeline.png"><img src="{stem}-search.png"><img src="{stem}-latents.png"></div>')
                gallery.append('</div>');selected.append(dict(stage=stage,comparison=label,category=category,**case))
    for name,rows in [('paired-cases',pairrows),('episodes',episode_rows)]:
        with (out/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'representative-cases.json').write_text(json.dumps(selected,indent=2))
    body='''<!doctype html><meta charset="utf-8"><title>G8/G9/G10 search evidence</title>
<style>body{font:16px system-ui;margin:30px;max-width:1600px}img{max-width:100%}video{width:320px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}td,th{padding:7px;border-bottom:1px solid #ddd}</style>
<h1>G8 fixed budgets / G9 warm start / G10 policy-initialized iCEM</h1>
<p>Official goal switching and tail-min scoring throughout. No pilot. G1/G7 development controls are explicitly reused;
all new configurations share their GPU/dtype, worker and video settings. Shared-node timing is not isolated GPU profiling.</p>
<p>G10-G8-96 is the main 96-trajectory comparison. G10 also projects all search actions to executable bounds as specified;
G8-96 retains the baseline policy/WM treatment. Equal trajectory budgets do not imply equal time or policy calls.</p>
<p>G10 follows the <a href="https://arxiv.org/abs/2008.06389">iCEM paper</a> and
<a href="https://github.com/martius-lab/iCEM/blob/98c1c1fe2bfc94a87e4658b6c793f4e81bc30203/icem/controllers/icem.py">pinned official implementation</a>,
with this task's policy initialization, conditional tails, zero-DC Torch noise and fixed 48/32/16 budget.</p>
<h2>Development</h2>'''+table(summary['full'])+'<img src="development.png"><img src="budget-and-time.png">'
    body+=f'<h2>New fixed-scene validation</h2><p>Locked method: {summary["selection"]["variant"]}; completed groups {len(summary["heldout"])}/3. This stage is separate from development; no outcome-based retuning.</p>'
    if summary['heldout']:body+=table(summary['heldout'])+'<img src="heldout.png">'
    body+='<p><a href="../run.json">Frozen protocol/reference hashes</a> | <a href="../selection.json">Selection lock</a> | <a href="../summary.json">All statistics</a> | <a href="paired-cases.csv">Paired cases</a> | <a href="episodes.csv">Episodes and censored transitions</a></p>'
    body+='<p>Examples are deterministic largest paired gain/loss, not manual behavior annotations. Lower imagined costs alone do not establish task benefit.</p>'+''.join(gallery)
    (out/'index.html').write_text(body,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);build_report(p.parse_args().root)
