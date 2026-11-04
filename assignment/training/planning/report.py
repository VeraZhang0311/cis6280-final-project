"""Offline B1 evidence report from steps.jsonl. No checkpoints, simulation or GPU required."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys
import warnings

import numpy as np

from assignment.training.planning.diagnostics import file_identity, write_json


def derive_steps(rows, goals):
    def values(key):
        return np.array([r.get(key) if r.get(key) is not None else np.nan for r in rows], float)
    data = {key: values(key) for key in ('sim_time', 'stacked_count', 'max_stacked_count',
                                        'active_goal_index', 'goal_distance_sq', 'goal_threshold_sq',
                                        'prediction_mse', 'plan_change_rms', 'decision_s', 'goal_dwell_s')}
    threshold = data['goal_threshold_sq']
    data['distance_ratio'] = np.divide(data['goal_distance_sq'], threshold,
                                      out=np.full(len(rows), np.nan), where=threshold > 0)
    data['goal_fraction'] = data['active_goal_index']/max(goals-1, 1)
    scores = [sorted(r.get('candidate_scores') or []) for r in rows]
    for name, getter in [('candidate_best', lambda s: s[0]), ('candidate_median', np.median),
                         ('candidate_worst', lambda s: s[-1]),
                         ('relative_gap', lambda s: (s[1]-s[0])/(abs(s[0])+1e-8) if len(s)>1 else np.nan)]:
        data[name] = np.array([getter(s) if s else np.nan for s in scores], float)
    return data


def finite_stat(values, percentile=50):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if len(values) else None


def load_episodes(root):
    episodes = []
    for path in sorted((Path(root)/'B1/episodes').glob('*/summary.json')):
        folder = path.parent
        summary = json.loads(path.read_text(encoding='utf-8'))
        rows = [json.loads(s) for s in (folder/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
        events = [json.loads(s) for s in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        data = derive_steps(rows, summary['goals'])
        first = summary['first_placement_step']
        after = data['sim_time'] >= first*.1 if first is not None else np.zeros(len(rows), bool)
        nonfinal = data['active_goal_index'] < summary['goals']-1
        metric = dict(scene=summary['scene_seed'], sample_seed=summary['sample_seed'],
                      peak=summary['max_placed'], final=summary['final_count'], success=summary['success'],
                      first_box_s=first*.1 if first is not None else None,
                      first_to_second_s=summary['first_to_second_s'],
                      post_first_observed_s=summary['post_first_observed_s'],
                      longest_nonfinal_goal_s=finite_stat(data['goal_dwell_s'][nonfinal], 100),
                      distance_ratio_median=finite_stat(data['distance_ratio']),
                      prediction_mse_p95=finite_stat(data['prediction_mse'], 95),
                      prediction_mse_after_first_median=finite_stat(data['prediction_mse'][after]),
                      plan_change_rms_median=finite_stat(data['plan_change_rms']),
                      plan_change_after_first_median=finite_stat(data['plan_change_rms'][after]),
                      candidate_relative_gap_median=finite_stat(data['relative_gap']),
                      decision_ms_mean=float(np.mean(data['decision_s'])*1000),
                      goal_jump_max=int(max([data['active_goal_index'][0], *np.diff(data['active_goal_index'])])),
                      final_goal_index=summary['goal_index'], goals=summary['goals'],
                      stack_count_drop_events=sum(e['event']=='stack_count_drop' for e in events),
                      zero_goal_threshold_blocks=int((data['goal_threshold_sq']==0).sum()),
                      episode=folder.name)
        tail_path = folder/'tail-state.json'
        if tail_path.exists():
            tail = json.loads(tail_path.read_text(encoding='utf-8'))[-1]
            metric['terminal_hand_contact_objects'] = np.flatnonzero(tail['object_hand_contacts']).tolist()
            metric['terminal_tower_objects'] = np.flatnonzero(tail['tower_members']).tolist()
        episodes.append(dict(folder=folder, summary=summary, rows=rows, events=events, data=data, metric=metric))
    return sorted(episodes, key=lambda e: (e['metric']['scene'], e['metric']['sample_seed']))


def verify_episodes(episodes, manifest, verify_videos=False):
    """Cross-check independently saved artifacts before interpreting them."""
    blocks = scores = videos = tails = 0
    options = manifest.get('evidence', {})
    for e in episodes:
        rows, summary, folder = e['rows'], e['summary'], e['folder']
        if [r['control_step'] for r in rows] != list(range(0, summary['steps'], 5)):
            raise ValueError(f'{folder.name}: misaligned block records')
        if rows[0]['prediction_mse'] is not None or rows[0]['plan_change_rms'] is not None:
            raise ValueError(f'{folder.name}: first prediction/plan comparison must be null')
        previous_goal = -1
        for r in rows:
            if abs(r['sim_time']-r['control_step']*.1) > 1e-8:
                raise ValueError(f'{folder.name}: simulation timestamp mismatch')
            if not previous_goal <= r['active_goal_index'] < summary['goals']:
                raise ValueError(f'{folder.name}: invalid goal progression')
            previous_goal = r['active_goal_index']
            values = r['candidate_scores']
            if len(values) != 16 or not np.isfinite(values).all():
                raise ValueError(f'{folder.name}: missing/nonfinite candidate scores')
            if r['selected_candidate'] != int(np.argmin(values)):
                raise ValueError(f'{folder.name}: selected candidate does not match baseline scoring')
            if r['control_step'] and (r['prediction_mse'] is None or r['plan_change_rms'] is None):
                raise ValueError(f'{folder.name}: missing one-block diagnostic')
            if not 0 <= r['stacked_count'] <= r['max_stacked_count'] <= summary['max_placed']:
                raise ValueError(f'{folder.name}: inconsistent progress counts')
            scores += len(values)
        terminal = e['events'][-1]
        if terminal['event'] != 'episode_end' or terminal['control_step'] != summary['steps'] \
                or terminal['final_count'] != summary['final_count']:
            raise ValueError(f'{folder.name}: terminal event / summary mismatch')
        if options.get('tail_state_steps', 0):
            tail = json.loads((folder/'tail-state.json').read_text(encoding='utf-8'))
            expected = list(range(max(1, summary['steps']-options['tail_state_steps']+1), summary['steps']+1))
            if [r['control_step'] for r in tail] != expected:
                raise ValueError(f'{folder.name}: incomplete final contact window')
            last = tail[-1]
            if last['count'] != summary['final_count'] or sum(last['tower_members']) != last['count']:
                raise ValueError(f'{folder.name}: terminal support count mismatch')
            if any(hand and member for hand, member in zip(last['object_hand_contacts'], last['tower_members'])):
                raise ValueError(f'{folder.name}: hand-contact object included in final tower')
            tails += len(tail)
        if options.get('keyframes', 0):
            if not (folder/'keyframes'/f'step-{summary["steps"]}.png').exists():
                raise ValueError(f'{folder.name}: terminal keyframe missing')
        if verify_videos:
            import imageio.v2 as imageio
            reader = imageio.get_reader(folder/'video.mp4')
            try:
                frames = reader.count_frames()
            finally:
                reader.close()
            expected_frames = len(rows)+(1 if options else 0)
            if frames != expected_frames:
                raise ValueError(f'{folder.name}: video has {frames} frames; expected {expected_frames}')
            videos += 1
        blocks += len(rows)
    return dict(episodes=len(episodes), block_records=blocks, candidate_scores=scores,
                tail_control_steps=tails, videos_decoded=videos, checks='passed')


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'savefig.dpi': 160})
    return plt


def sampled(episode, key, grid, align_first=False):
    times = episode['data']['sim_time']
    query = grid.copy()
    if align_first:
        first = episode['metric']['first_box_s']
        if first is None:
            return np.full(len(grid), np.nan)
        query += first
    index = np.clip(np.searchsorted(times, query), 0, len(times)-1)
    before = np.maximum(index-1, 0)
    index = np.where(abs(times[before]-query) < abs(times[index]-query), before, index)
    valid = (query >= times[0]) & (query <= times[-1]) & (abs(times[index]-query) <= .250001)
    return np.where(valid, episode['data'][key][index], np.nan)


def band(ax, grid, matrix, label, color='#2673a6', log=False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        q1, med, q3 = np.nanpercentile(matrix, [25, 50, 75], axis=0)
    ax.fill_between(grid, q1, q3, color=color, alpha=.18, linewidth=0)
    ax.plot(grid, med, color=color, label=label)
    if log and np.any(matrix > 0):
        ax.set_yscale('symlog', linthresh=1e-6)


def plot_overview(episodes, out):
    plt = plotting()
    scenes = sorted({e['metric']['scene'] for e in episodes})
    seeds = sorted({e['metric']['sample_seed'] for e in episodes})
    fig, axes = plt.subplots(1, 2, figsize=(9, max(6, len(scenes)*.32)), constrained_layout=True)
    for ax, key, title in zip(axes, ['peak', 'final'], ['Peak counted boxes', 'Final counted boxes']):
        matrix = np.full((len(scenes), len(seeds)), np.nan)
        for episode in episodes:
            m = episode['metric']
            matrix[scenes.index(m['scene']), seeds.index(m['sample_seed'])] = m[key]
        im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=4, aspect='auto')
        for i in range(len(scenes)):
            for j in range(len(seeds)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, str(int(matrix[i, j])), ha='center', va='center',
                            color='white' if matrix[i, j] >= 3 else '#182b39')
        ax.set(xticks=range(len(seeds)), xticklabels=seeds, yticks=range(len(scenes)), yticklabels=scenes,
               xlabel='Policy sampler seed', ylabel='Validation scene', title=title)
        ax.grid(False)
    fig.colorbar(im, ax=axes, ticks=range(5), shrink=.65, label='Boxes (official instantaneous count)')
    fig.suptitle(f'B1 | {len(scenes)} fixed scenes, {len(episodes)} episodes')
    fig.savefig(out/'outcomes.png')
    plt.close(fig)


def plot_aggregate(episodes, out, aligned=False):
    plt = plotting()
    selected = [e for e in episodes if e['metric']['first_box_s'] is not None] if aligned else episodes
    grid = np.arange(-20, 120.01, .5) if aligned else np.arange(0, 179.01, .5)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    specs = [('stacked_count', 'Current counted boxes', False),
             ('distance_ratio', 'Goal distance / switch threshold', True),
             ('prediction_mse', 'One-block latent prediction MSE', True),
             ('plan_change_rms', 'Replanned first block vs previous second block (RMS)', False),
             ('relative_gap', '(Second-best score - best) / |best|', True),
             ('goal_fraction', 'Active goal index / (number of goals - 1)', False)]
    for ax, (key, title, log) in zip(axes.flat, specs):
        matrix = np.array([sampled(e, key, grid, aligned) for e in selected])
        band(ax, grid, matrix, 'Median; shaded IQR', log=log)
        ax.set_title(title)
        if key == 'distance_ratio':
            ax.axhline(1, color='#ce7e27', ls='--', lw=1, label='Switch threshold = 1')
        if key == 'stacked_count':
            ax.set_ylim(-.1, 4.1)
        if key == 'goal_fraction':
            ax.set_ylim(-.03, 1.03)
        if aligned:
            ax.axvline(0, color='#5b8668', alpha=.7, lw=1)
        ax.legend(fontsize=8, loc='best')
    xlabel = 'Seconds relative to first counted box (nearest real block, within 0.25 s)' if aligned else 'Simulation time (s)'
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(f'B1 steps.jsonl | {len(selected)} episodes | descriptive medians, no causal interpretation')
    fig.savefig(out/('after-first-box.png' if aligned else 'steps-aggregate.png'))
    plt.close(fig)
    if aligned:
        fig, ax = plt.subplots(figsize=(10, 2.5), constrained_layout=True)
        matrix = np.array([sampled(e, 'stacked_count', grid, True) for e in selected])
        ax.step(grid, np.isfinite(matrix).sum(axis=0), where='mid', color='#2673a6')
        ax.set(xlabel='Seconds relative to first counted box', ylabel='Observed episodes',
               title='Episodes contributing at each aligned time (missing observations excluded)')
        fig.savefig(out/'after-first-observation-count.png')
        plt.close(fig)


def plot_heatmaps(episodes, out):
    plt = plotting()
    grid = np.arange(0, 179.01, .5)
    labels = [f'{e["metric"]["scene"]} / {e["metric"]["sample_seed"]}' for e in episodes]
    groups = [('progress-goals', [('stacked_count', 'Current count', False),
                                  ('goal_fraction', 'Active goal fraction', False),
                                  ('distance_ratio', 'Goal distance / threshold', True)]),
              ('prediction-plan-candidates', [('prediction_mse', 'One-block latent MSE', True),
                                              ('plan_change_rms', 'Plan change RMS', False),
                                              ('relative_gap', 'Relative best / runner-up gap', True)])]
    for name, specs in groups:
        fig, axes = plt.subplots(1, 3, figsize=(18, max(9, len(episodes)*.24)), sharey=True, constrained_layout=True)
        for ax, (key, title, log) in zip(axes, specs):
            data = np.array([sampled(e, key, grid) for e in episodes])
            if log:
                data = np.log10(np.maximum(data, 1e-12))
            finite = data[np.isfinite(data)]
            low, high = np.percentile(finite, [1, 99]) if len(finite) else (0, 1)
            if low == high:
                high = low+1
            if key == 'stacked_count':
                low, high = 0, 4
            if key == 'goal_fraction':
                low, high = 0, 1
            cmap = plt.get_cmap('viridis').copy()
            cmap.set_bad('#dddddd')
            im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=low, vmax=high,
                           extent=[-.25, 179.25, len(episodes)-.5, -.5], interpolation='nearest')
            ax.set(title=title, xlabel='Simulation time (s)', yticks=range(len(labels)), yticklabels=labels)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(False)
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=.015, shrink=.9,
                         label=('log10 value; color clipped at P1/P99' if log else 'Value'), extend='both')
        axes[0].set_ylabel('Scene / sampler seed')
        fig.suptitle('B1 steps.jsonl | gray = no observation / undefined value')
        fig.savefig(out/f'{name}-heatmap.png')
        plt.close(fig)


def choose_representatives(episodes):
    choices = []
    rules = [('highest peak', lambda e: (e['metric']['peak'], e['metric']['final'])),
             ('lowest peak', lambda e: (-e['metric']['peak'], -e['metric']['final'])),
             ('longest nonfinal goal residence', lambda e: e['metric']['longest_nonfinal_goal_s'] or 0),
             ('highest prediction-error P95', lambda e: e['metric']['prediction_mse_p95'] or 0),
             ('largest median plan change', lambda e: e['metric']['plan_change_rms_median'] or 0),
             ('smallest candidate gap', lambda e: -(e['metric']['candidate_relative_gap_median'] or 0))]
    for reason, key in rules:
        used = {c[1]['folder'].name for c in choices}
        remaining = [e for e in episodes if e['folder'].name not in used]
        if remaining:
            choices.append((reason, max(remaining, key=key)))
    return choices


def plot_task_progress(episodes, out):
    plt = plotting()
    control_steps = np.arange(1796)
    grid = control_steps*.1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for k, color in zip(range(1, 5), ['#2673a6', '#ce7e27', '#5b8668', '#8a63a0']):
        reached = []
        for e in episodes:
            stages = e['summary']['stage_steps']
            reached.append(control_steps >= stages[k-1] if len(stages) >= k else np.zeros(len(grid), bool))
        axes[0].plot(grid, np.mean(reached, axis=0), label=f'Ever counted >= {k}', color=color)
    axes[0].set(xlabel='Simulation time (s)', ylabel='Fraction of all episodes', ylim=(0, 1.03),
                title='Exact control-step progress events')
    axes[0].legend(fontsize=8)
    bins = np.arange(5)
    for offset, key, label, color in [(-.18, 'peak', 'Peak', '#2673a6'), (.18, 'final', 'Final', '#ce7e27')]:
        counts = [sum(e['metric'][key]==k for e in episodes) for k in bins]
        axes[1].bar(bins+offset, counts, width=.35, label=label, color=color)
    axes[1].set(xticks=bins, xlabel='Counted boxes', ylabel='Episodes', title='Peak and terminal counts')
    axes[1].legend()
    fig.savefig(out/'task-progress.png')
    plt.close(fig)
    first = [e for e in episodes if e['metric']['first_box_s'] is not None]
    if not first:
        return
    first.sort(key=lambda e: (e['metric']['first_to_second_s'] is None,
                              e['metric']['first_to_second_s'] if e['metric']['first_to_second_s'] is not None
                              else e['metric']['post_first_observed_s']))
    fig, ax = plt.subplots(figsize=(12, max(5, len(first)*.24)), constrained_layout=True)
    for i, e in enumerate(first):
        m = e['metric']
        reached = m['first_to_second_s'] is not None
        end = m['first_to_second_s'] if reached else m['post_first_observed_s']
        color = '#2673a6' if reached else '#ce7e27'
        ax.hlines(i, 0, end, color=color, lw=2.5, alpha=.6)
        ax.scatter(end, i, color=color, marker='o' if reached else '|', s=45)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color='#2673a6', marker='o', label='Second box reached'),
                       Line2D([], [], color='#ce7e27', marker='|', label='No second box: observation ended')])
    ax.set(yticks=range(len(first)),
           yticklabels=[f'{e["metric"]["scene"]} / {e["metric"]["sample_seed"]}' for e in first],
           xlabel='Seconds after first counted box', ylabel='Scene / sampler seed',
           title='First-to-second-box transitions and right-censored observations')
    ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()
    fig.savefig(out/'first-to-second.png')
    plt.close(fig)


def plot_detail(episode, path, reason):
    plt = plotting()
    d, s = episode['data'], episode['summary']
    t = d['sim_time']
    fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True, constrained_layout=True)
    axes[0].step(t, d['stacked_count'], where='post', label='Current count')
    axes[0].step(t, d['max_stacked_count'], where='post', color='#ce7e27', label='Peak count')
    axes[0].scatter(s['steps']*.1, s['final_count'], marker='X', color='#9c3f3b', s=65, label='Terminal count')
    axes[0].set(ylabel='Boxes', ylim=(-.15, 4.2))
    axes[0].legend(loc='upper left', ncols=3)
    axes[1].plot(t, d['goal_distance_sq'], label='Current goal squared distance')
    axes[1].plot(t, d['goal_threshold_sq'], color='#ce7e27', label='Switch threshold')
    twin = axes[1].twinx()
    twin.step(t, d['active_goal_index'], where='post', color='#686868', alpha=.6)
    twin.set_ylabel('Active goal index')
    axes[1].set_ylabel('Squared distance')
    axes[1].legend(loc='upper left', fontsize=8)
    axes[2].fill_between(t, d['candidate_best'], d['candidate_worst'], alpha=.15, color='#2673a6')
    axes[2].plot(t, d['candidate_best'], label='Best')
    axes[2].plot(t, d['candidate_median'], color='#ce7e27', label='Median')
    axes[2].set(ylabel='Candidate score', title='Predicted 8 s endpoint cost (lower is better); shaded min-max')
    axes[2].legend(fontsize=8)
    axes[3].plot(t, d['prediction_mse'])
    axes[3].set(ylabel='Next-latent MSE')
    if np.any(d['prediction_mse'] > 0):
        axes[3].set_yscale('log')
    axes[4].plot(t, d['plan_change_rms'])
    axes[4].set_ylabel('Plan change RMS')
    axes[5].plot(t, 100*d['relative_gap'], label='Relative runner-up gap (%)')
    axes[5].set_ylabel('Relative gap (%)')
    twin = axes[5].twinx()
    twin.plot(t, 1000*d['decision_s'], color='#686868', alpha=.45)
    twin.set_ylabel('Decision time (ms)')
    for event in episode['events']:
        if event['event'] in ('first_box', 'second_box'):
            for ax in axes:
                ax.axvline(event['sim_time'], color='#5b8668', alpha=.45, lw=1)
            axes[0].annotate(event['event'], (event['sim_time'], event['stacked_count']),
                             xytext=(3, 8), textcoords='offset points', fontsize=8)
    axes[-1].set_xlabel('Simulation time (s)')
    fig.suptitle(f'B1 / {episode["folder"].name} | {reason} | peak {s["max_placed"]}, final {s["final_count"]}')
    fig.savefig(path)
    plt.close(fig)


def write_browser(episodes, out, summary, representatives):
    """Offline index links each selected episode's existing video, figure, and logs."""
    metrics = [e['metric'] for e in episodes]
    scenes = sorted({m['scene'] for m in metrics})
    seeds = sorted({m['sample_seed'] for m in metrics})
    table_rows = []
    for m in metrics:
        path = '../B1/episodes/'+m['episode']
        table_rows.append(f'<tr><td>{m["scene"]}</td><td>{m["sample_seed"]}</td>'
                          f'<td>{m["peak"]}</td><td>{m["final"]}</td>'
                          f'<td>{m["longest_nonfinal_goal_s"]}</td>'
                          f'<td><a href="{path}/video.mp4">Video</a> · '
                          f'<a href="{path}/steps.jsonl">Steps</a> · '
                          f'<a href="{path}/tail-state.json">Contacts/support</a></td></tr>')
    selected = {e['folder'].name: e['metric'] for e in episodes}
    detail = {e['folder'].name: f'detail-{e["folder"].name}.png' for e in episodes}
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>B1 evidence report</title>
<style>body{font:15px system-ui,sans-serif;color:#243442;background:#fafbfc;margin:24px auto;max-width:1400px;padding:0 20px}
h1,h2{font-weight:600}a{color:#176a9a}img{max-width:100%;height:auto}select{font:inherit;padding:6px;margin:8px}
.row{display:flex;gap:24px;align-items:flex-start}.row video{width:320px;max-width:42vw}.row img{width:850px;max-width:70vw}
table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #dae0e6;text-align:left}
.note{line-height:1.6;max-width:1000px}code{background:#eef1f4;padding:2px 5px}
@media(max-width:700px){.row{display:block}.row video,.row img{max-width:100%;width:100%}}</style>
<h1>B1 / 20-scene diagnostic evidence</h1><p class="note">SUMMARY</p>
<p class="note">Descriptive results on fixed validation scenes and sampler seeds 0, 2, 3.
Final count excludes boxes touching the hand. Count drops do not establish physical collapse.
The next-latent MSE compares the previous one-block prediction with the current observation;
candidate scores predict an 8-second endpoint. These are different horizons.</p>
<h2>Episode explorer</h2><label>Scene <select id="scene">SCENES</select></label>
<label>Sampler seed <select id="seed">SEEDS</select></label><p id="selected"></p>
<div class="row"><video id="video" controls preload="none"></video><a id="figureLink"><img id="figure" alt="Selected episode evidence timeline"></a></div>
<p><a id="steps">steps.jsonl</a> · <a id="events">events.jsonl</a> · <a id="tail">tail-state.json</a></p>
<h2>Outcomes by scene and sampler seed</h2><a href="outcomes.png"><img src="outcomes.png" alt="Peak and final box counts"></a>
<a href="task-progress.png"><img src="task-progress.png" alt="Cumulative task progress and terminal counts"></a>
<h2>Block-level evidence</h2><a href="steps-aggregate.png"><img src="steps-aggregate.png" alt="Median and IQR of planner diagnostics"></a>
<p><a href="progress-goals-heatmap.png">All episodes: progress and goals</a> ·
<a href="prediction-plan-candidates-heatmap.png">All episodes: prediction, plans, candidate gaps</a></p>
<h2>Aligned to the first counted box</h2><a href="after-first-box.png"><img src="after-first-box.png" alt="Diagnostics aligned to first box"></a>
<img src="after-first-observation-count.png" alt="Number of observed episodes by aligned time">
<p><a href="first-to-second.png">First-to-second transitions including right-censored observations</a></p>
<p class="note">Bands show descriptive medians and IQR. Missing observations are excluded, including episodes
without a first box in the aligned plots. The contributing-episode count changes over time; successful
first-to-second durations exclude censored episodes. These plots do not identify failure causes.</p>
<h2>All B1 episodes</h2><p><a href="episode-metrics.csv">Metrics CSV</a> · <a href="episode-metrics.json">Metrics JSON</a> · <a href="REPORT.md">Analysis notes</a></p>
<table><thead><tr><th>Scene</th><th>Seed</th><th>Peak</th><th>Final</th><th>Longest nonfinal goal (s)</th><th>Evidence</th></tr></thead><tbody>ROWS</tbody></table>
<script>const episodes=EPISODES;const details=DETAILS;
const scene=document.getElementById('scene'),seed=document.getElementById('seed');
function update(){const name='scene-'+scene.value+'.sample-'+seed.value,m=episodes[name];if(!m)return;
const base='../B1/episodes/'+name+'/';document.getElementById('video').src=base+'video.mp4';
const image=details[name]||base+'timeline.png';document.getElementById('figure').src=image;
document.getElementById('figureLink').href=image;document.getElementById('selected').textContent=
'Peak '+m.peak+' / final '+m.final+'; longest nonfinal goal '+m.longest_nonfinal_goal_s+' s';
document.getElementById('steps').href=base+'steps.jsonl';document.getElementById('events').href=base+'events.jsonl';
document.getElementById('tail').href=base+'tail-state.json';}
scene.addEventListener('change',update);seed.addEventListener('change',update);update();</script></html>'''
    document = document.replace('SUMMARY', html.escape(summary))
    document = document.replace('20-scene', f'{len(scenes)}-scene')
    document = document.replace('SCENES', ''.join(f'<option>{s}</option>' for s in scenes))
    document = document.replace('SEEDS', ''.join(f'<option>{s}</option>' for s in seeds))
    document = document.replace('ROWS', ''.join(table_rows))
    document = document.replace('EPISODES', json.dumps(selected, allow_nan=False))
    document = document.replace('DETAILS', json.dumps(detail))
    (out/'index.html').write_text(document, encoding='utf-8')


def build_report(root, allow_partial=False, verify_videos=False):
    root = Path(root).resolve()
    manifest = json.loads((root/'run.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' and not allow_partial:
        raise ValueError('Run is not complete; use --allow-partial for a clearly labeled preliminary report')
    episodes = load_episodes(root)
    if not episodes:
        raise ValueError('No completed B1 episodes found')
    expected = len(manifest['scene_seeds'])*len(manifest['sample_seeds'])
    complete = len(episodes) == expected and manifest['status'] == 'complete'
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    verification = verify_episodes(episodes, manifest, verify_videos)
    write_json(out/'verification.json', verification)
    metrics = [e['metric'] for e in episodes]
    write_json(out/'episode-metrics.json', metrics)
    fields = sorted({k for m in metrics for k in m})
    with (out/'episode-metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    plot_overview(episodes, out)
    plot_task_progress(episodes, out)
    plot_aggregate(episodes, out)
    if any(m['first_box_s'] is not None for m in metrics):
        plot_aggregate(episodes, out, aligned=True)
    plot_heatmaps(episodes, out)
    representatives = choose_representatives(episodes)
    reasons = {e['folder'].name: reason for reason, e in representatives}
    for episode in episodes:
        plot_detail(episode, out/f'detail-{episode["folder"].name}.png',
                    reasons.get(episode['folder'].name, 'block-level diagnostic evidence'))
    write_json(out/'representatives.json', [dict(reason=reason, **e['metric']) for reason, e in representatives])
    first = [m for m in metrics if m['first_box_s'] is not None]
    second = [m for m in first if m['first_to_second_s'] is not None]
    aggregate = dict(complete=complete, episodes=len(episodes), expected_episodes=expected,
                     scenes=len({m['scene'] for m in metrics}),
                     reached={str(k): sum(m['peak']>=k for m in metrics)/len(metrics) for k in range(1, 5)},
                     mean_peak=float(np.mean([m['peak'] for m in metrics])),
                     mean_final=float(np.mean([m['final'] for m in metrics])),
                     success_rate=float(np.mean([m['success'] for m in metrics])),
                     first_box_episodes=len(first), second_given_first=len(second)/len(first) if first else None,
                     censored_after_first=len(first)-len(second),
                     longest_nonfinal_goal_s=finite_stat([m['longest_nonfinal_goal_s'] for m in metrics], 100),
                     episodes_nonfinal_goal_over_60s=sum((m['longest_nonfinal_goal_s'] or 0)>60 for m in metrics),
                     goal_skipping_episodes=sum(m['goal_jump_max']>1 for m in metrics),
                     final_less_than_peak_episodes=sum(m['final']<m['peak'] for m in metrics))
    gaps = np.concatenate([e['data']['relative_gap'] for e in episodes])
    gaps = gaps[np.isfinite(gaps)]
    aggregate['candidate_relative_gap_median'] = finite_stat(gaps)
    aggregate['candidate_gap_below_1pct_fraction'] = float(np.mean(gaps < .01)) if len(gaps) else None
    aggregate['prediction_mse_median'] = finite_stat(np.concatenate([e['data']['prediction_mse'] for e in episodes]))
    aggregate['plan_change_rms_median'] = finite_stat(np.concatenate([e['data']['plan_change_rms'] for e in episodes]))
    write_json(out/'aggregate.json', aggregate)
    text = f'{"COMPLETE" if complete else "PARTIAL"}: {len(episodes)}/{expected} episodes; '
    text += f'mean peak {aggregate["mean_peak"]:.3f}; mean final {aggregate["mean_final"]:.3f}.'
    write_browser(episodes, out, text, representatives)
    lines = ['# B1 批量诊断证据', '', text, '',
             '所有比例均为本次固定场景/seed 的描述性统计；同一 scene 的三个 sampler seeds 不作为独立场景推断。', '',
             '| 指标 | 结果 |', '| --- | --- |']
    for k in range(1, 5):
        lines.append(f'| 曾达到至少 {k} 箱 | {aggregate["reached"][str(k)]:.1%} |')
    lines.extend([f'| 第一箱 episode 数 | {len(first)} |',
                  f'| 第一箱之后达到第二箱 | {len(second)}/{len(first)} |',
                  f'| 第一箱之后未到第二箱（截尾） | {len(first)-len(second)} |',
                  f'| 非末尾子目标停留超过 60 秒的 episode 数 | {aggregate["episodes_nonfinal_goal_over_60s"]} |',
                  f'| 曾一次跳过多个子目标的 episode 数 | {aggregate["goal_skipping_episodes"]} |',
                  f'| final < peak 的 episode 数 | {aggregate["final_less_than_peak_episodes"]} |', '',
                  '## steps.jsonl 可视化的定义', '',
                  '- `steps-aggregate.png`：六种信号按仿真时间汇总的中位数与四分位区间。',
                  '- `task-progress.png` 从每个 control step 的 stage_steps 画累计到达比例；`first-to-second.png` 同时显示成功过渡与未到第二箱的实际观测窗口。',
                  '- 两张 heatmap：逐 episode、逐 block 的值；灰色表示无观测或量未定义。log10 色条按 P1/P99 裁色，零值仅在着色时截到 1e-12，原始数值保留。汇总对数图在 1e-6 以下使用线性刻度，以显示零值。',
                  '- `after-first-box.png`：仅纳入达到第一箱的 episodes，按首次计数时间对齐，匹配最近 0.5 秒观测点（误差不超过 0.25 秒）。另图显示实际可观测分母。',
                  '- 目标距离比值 = goal_distance_sq / goal_threshold_sq，阈值为零时记缺失；比值 < 1 满足距离切换条件，最后目标仍不会再推进。',
                  '- 目标字段是在本轮完成切换后记录的，因此非末尾目标比值通常 >= 1 是算法预期；这个事实本身不能证明阈值不合理。',
                  '- relative gap = (第二低候选分数 - 最低分数) / (|最低分数| + 1e-8)，只说明模型评分区分度。',
                  '- MSE 是上一轮一个 block 的预测对本轮真实编码；候选分数是八秒预测终点，不能混为同一 horizon 的误差。',
                  '- MSE 是整体预测—执行一致性，包含采样动作与实际裁剪动作的差异，不能单独归因于 world model。',
                  '- 计划 RMS 使用 raw/clipped action 单位，比较上一轮计划第二个 block 与本轮第一个 block。',
                  '- 决策耗时含诊断提取与 GPU 同步；本批 4 workers 共享一张 L40，包含并发争用，不作为单网络性能基准。',
                  '- long goal residence 排除最后一个目标，避免把目标序列走完后的停留误称为目标卡住。', '',
                  '## 代表案例（按可观察指标选取，不是因果分类）', ''])
    for reason, e in representatives:
        name = e['folder'].name
        lines.append(f'- {reason}: [{name}](detail-{name}.png)，peak/final={e["metric"]["peak"]}/{e["metric"]["final"]}。')
    lines.extend(['', '计数下降不直接证明物理倒塌；final_count 会排除仍有手接触的箱子。候选评分差异不证明有效候选覆盖，预测/计划变化也不单独证明失败原因。', '',
                  '本批次保存完整视频、逐 block 标量/候选评分、逐 control-step 任务事件、末尾 56 步物体/接触/支撑状态，以及末尾六张关键帧。没有常规保存全部候选动作或 latent 张量。'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    write_json(out/'provenance.json', dict(report_source=file_identity(__file__),
                                          software=dict(python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__),
                                          run=file_identity(root/'run.json'),
                                          inputs=[file_identity(e['folder']/'steps.jsonl') for e in episodes]))
    print(json.dumps(aggregate, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--verify-videos', action='store_true', help='Decode/count every video to verify terminal-frame coverage')
    args = p.parse_args()
    build_report(args.run, args.allow_partial, args.verify_videos)


if __name__ == '__main__':
    main()
