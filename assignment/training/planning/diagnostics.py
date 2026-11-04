"""Phase-one recording and offline analysis. Never feeds simulator state to an agent."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CONTROL_DT = .1


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


class EpisodeDiagnostics:
    def __init__(self):
        self.steps, self.events = [], []
        self.count = self.peak = 0
        self.first = self.second = None
        self.goal = None
        self.goal_since = 0

    def event(self, name, step, **fields):
        self.events.append(dict(event=name, control_step=step, sim_time=step*CONTROL_DT, **fields))

    def decision(self, step, diagnostic, elapsed):
        index = diagnostic['active_goal_index']
        if index != self.goal:
            self.event('goal_changed', step, previous_index=self.goal, active_goal_index=index)
            self.goal, self.goal_since = index, step
        row = dict(control_step=step, sim_time=step*CONTROL_DT, stacked_count=self.count,
                   max_stacked_count=self.peak, **diagnostic,
                   goal_dwell_s=(step-self.goal_since)*CONTROL_DT if index is not None else None,
                   decision_s=elapsed)
        self.steps.append(row)
        return row

    def task_step(self, step, count):
        # Called at EVERY control step, including events between rendered observations.
        count = int(count)
        if self.first is None and count >= 1:
            self.first = step
            self.event('first_box', step, stacked_count=count)
        if self.second is None and count >= 2:
            self.second = step
            self.event('second_box', step, stacked_count=count)
        if count < self.count:
            self.event('stack_count_drop', step, previous_count=self.count, stacked_count=count)
        self.count, self.peak = count, max(self.peak, count)

    def finish(self, result):
        step = result['steps']
        self.event('episode_end', step, success=result['success'], final_count=result['final_count'],
                   max_placed=result['max_placed'])
        return dict(second_placement_step=self.second,
                    first_to_second_s=(self.second-self.first)*CONTROL_DT if self.second is not None else None,
                    post_first_observed_s=(step-self.first)*CONTROL_DT if self.first is not None else None,
                    second_reached=self.second is not None,
                    decision_count=len(self.steps),
                    decision_s_mean=float(np.mean([s['decision_s'] for s in self.steps])),
                    decision_s_p95=float(np.percentile([s['decision_s'] for s in self.steps], 95)))

    def save(self, folder, result):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name, rows in [('steps', self.steps), ('events', self.events)]:
            (folder/f'{name}.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows),
                                              encoding='utf-8')
        write_json(folder/'summary.json', result)
        plot_timeline(self.steps, self.events, folder/'timeline.png')


def annotate_frame(frame, goal, row, label):
    from PIL import Image, ImageDraw
    # Keep the camera and goal images intact; put labels in separate strips.
    h, w = frame.shape[:2]
    canvas = Image.new('RGB', (w, 2*h+96), '#151b26')
    canvas.paste(Image.fromarray(frame), (0, 32))
    if goal is not None:
        canvas.paste(Image.fromarray(goal), (0, h+64))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 3), f'{label}  t={row["sim_time"]:.1f}s', fill='white')
    draw.text((5, 16), f'stack={row["stacked_count"]}  peak={row["max_stacked_count"]}  front | side', fill='white')
    text = f'active subgoal {row["active_goal_index"]}' if goal is not None else 'B0: no active subgoal'
    draw.text((5, h+40), text, fill='white')
    return np.asarray(canvas)


def plot_timeline(rows, events, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    path = Path(path)
    fig.suptitle(f'{path.parents[2].name} / {path.parent.name}')
    times = [r['sim_time'] for r in rows]
    axes[0].step(times, [r['stacked_count'] for r in rows], where='post', label='current')
    axes[0].step(times, [r['max_stacked_count'] for r in rows], where='post', label='peak')
    # Events preserve brief progress / drops between block observations.
    for event in events:
        if event['event'] in ('first_box', 'second_box', 'stack_count_drop'):
            axes[0].scatter(event['sim_time'], event['stacked_count'], s=20)
    axes[0].set_ylabel('Stacked boxes')
    axes[0].legend(loc='upper left')
    def series(key):
        return [r.get(key) if r.get(key) is not None else np.nan for r in rows]
    axes[1].set_ylabel('Goal distance')
    if any(r.get('active_goal_index') is not None for r in rows):
        axes[1].plot(times, series('goal_distance_sq'), label='distance squared')
        axes[1].plot(times, series('goal_threshold_sq'), label='switch threshold')
        twin = axes[1].twinx()
        twin.step(times, series('active_goal_index'), where='post', color='gray', alpha=.6)
        twin.set_ylabel('Goal index')
        axes[1].legend(loc='upper left')
    else:
        axes[1].text(.5, .5, 'Not applicable: B0 has no active subgoal',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_yticks([])
    if any(r.get('prediction_mse') is not None for r in rows):
        axes[2].plot(times, series('prediction_mse'))
    else:
        axes[2].text(.5, .5, 'Not available: no predictor rollout',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_yticks([])
    axes[2].set_ylabel('Next-latent MSE')
    axes[3].plot(times, series('plan_change_rms'))
    axes[3].set_ylabel('Plan change RMS')
    axes[3].set_xlabel('Simulation time (s)')
    for event in events:
        if event['event'] in ('first_box', 'second_box'):
            for ax in axes:
                ax.axvline(event['sim_time'], color='green', alpha=.35)
            axes[0].annotate(event['event'], (event['sim_time'], event['stacked_count']))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def aggregate(results, decision_times):
    from assignment.training.planning.evaluate import summarize
    summary = summarize(results, 4)
    first = [r for r in results if r['first_placement_step'] is not None]
    transitions = [r['first_to_second_s'] for r in results if r['first_to_second_s'] is not None]
    summary.update(mean_max_placed=float(np.mean([r['max_placed'] for r in results])),
                   mean_final_count=float(np.mean([r['final_count'] for r in results])),
                   first_box_episodes=len(first),
                   second_given_first=sum(r['second_reached'] for r in first)/len(first) if first else None,
                   first_to_second_completed_episodes=len(transitions),
                   first_to_second_s_mean=float(np.mean(transitions)) if transitions else None,
                   post_first_censored_episodes=sum(not r['second_reached'] for r in first),
                   decision_s_mean=float(np.mean(decision_times)) if decision_times else None,
                   decision_s_p95=float(np.percentile(decision_times, 95)) if decision_times else None)
    return summary


def analyze_run(root):
    """Rebuild figures, paired summaries and review tables from saved scalars only."""
    root = Path(root)
    by_label, times, sample_times, reviews, evidence = {}, {}, {}, [], []
    for path in sorted(root.glob('B*/episodes/*/summary.json')):
        folder, label = path.parent, path.parents[2].name
        result = json.loads(path.read_text(encoding='utf-8'))
        rows = [json.loads(s) for s in (folder/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
        events = [json.loads(s) for s in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        plot_timeline(rows, events, folder/'timeline.png')
        by_label.setdefault(label, []).append(result)
        times.setdefault(label, []).extend(r['decision_s'] for r in rows)
        sample_times.setdefault((label, result['sample_seed']), []).extend(r['decision_s'] for r in rows)
        after = [r for r in rows if result['first_placement_step'] is not None
                 and r['control_step'] >= result['first_placement_step']]
        scores = [sorted(r['candidate_scores']) for r in after if r['candidate_scores']]
        evidence.append(dict(baseline=label, scene=result['scene_seed'], sample_seed=result['sample_seed'],
                             peak=result['max_placed'], final=result['final_count'],
                             first_to_second_s=result['first_to_second_s'],
                             post_first_observed_s=result['post_first_observed_s'],
                             goal_indices_after_first=sorted({r['active_goal_index'] for r in after
                                                              if r['active_goal_index'] is not None}),
                             max_goal_dwell_s=max((r['goal_dwell_s'] for r in after
                                                   if r['goal_dwell_s'] is not None), default=None),
                             mean_best_runnerup_gap=float(np.mean([s[1]-s[0] for s in scores if len(s)>1]))
                             if any(len(s)>1 for s in scores) else None,
                             mean_prediction_mse=float(np.mean([r['prediction_mse'] for r in after
                                                                if r['prediction_mse'] is not None]))
                             if any(r['prediction_mse'] is not None for r in after) else None,
                             mean_plan_change_rms=float(np.mean([r['plan_change_rms'] for r in after
                                                                 if r['plan_change_rms'] is not None]))
                             if any(r['plan_change_rms'] is not None for r in after) else None,
                             episode=str(folder.relative_to(root))))
        reviews.append(dict(baseline=label, scene=result['scene_seed'], sample_seed=result['sample_seed'],
                            peak=result['max_placed'], final=result['final_count'],
                            episode=str(folder.relative_to(root)), approached_another_box='',
                            contacted_another_box='', lifted_another_box='', near_static_or_same_region='',
                            disturbed_existing_stack='', observed_behavior='', hypothesis='', next_check=''))
    summary = {label: aggregate(results, times[label]) for label, results in by_label.items()}
    for label, results in by_label.items():
        summary[label]['by_sample_seed'] = {
            str(seed): aggregate([r for r in results if r['sample_seed'] == seed], sample_times[label, seed])
            for seed in sorted({r['sample_seed'] for r in results})}
    write_json(root/'summary.json', summary)
    write_json(root/'evidence.json', evidence)
    # Never overwrite the user's video annotations on re-analysis.
    review_path = root/'review.csv'
    if reviews and not review_path.exists():
        with review_path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(reviews[0]))
            writer.writeheader()
            writer.writerows(reviews)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    print(json.dumps(analyze_run(args.run), indent=2))


if __name__ == '__main__':
    main()
