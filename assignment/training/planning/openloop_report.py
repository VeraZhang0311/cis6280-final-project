"""Offline evidence for one fixed 8-second plan after first placement."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from assignment.training.planning.diagnostics import file_identity, write_json
from assignment.training.planning.openloop_metrics import aligned_trajectory_metrics, candidate_goal_distances
from assignment.training.planning.report import plotting


def load_and_verify(root):
    manifest = json.loads((root/'run.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('measurement run has not completed')
    all_cases, valid = [], []
    for folder in sorted((root/'cases').iterdir()):
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        all_cases.append(summary)
        if summary['status'] != 'complete':
            continue
        data = dict(np.load(folder/'trajectories.npz'))
        rows = [json.loads(line) for line in (folder/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
        selected = summary['selected_candidate']
        if summary['model_calls']['test_rollout'] != 1 or summary['model_calls']['test_sample'] != 1:
            raise ValueError(f'{folder.name}: measurement requires exactly one sample and WM rollout')
        if data['imagined_latents'].shape[:2] != (16, 16) or data['observed_latents'].shape != (17, data['goal_latent'].size):
            raise ValueError(f'{folder.name}: invalid future/context latent alignment')
        if data['raw_action_blocks'].shape != (16, 16, 5, 25) or np.abs(data['raw_action_blocks']).max() > 1:
            raise ValueError(f'{folder.name}: invalid action blocks')
        if selected != int(data['baseline_terminal_scores'].argmin()):
            raise ValueError(f'{folder.name}: terminal selection changed')
        recomputed = aligned_trajectory_metrics(data['imagined_latents'][selected], data['observed_latents'][1:], data['goal_latent'])
        for key, values in recomputed.items():
            np.testing.assert_allclose(values, data[key], rtol=1e-7, atol=1e-10)
            np.testing.assert_allclose(values, [row[key] for row in rows], rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(candidate_goal_distances(data['imagined_latents'], data['goal_latent']),
                                   data['candidate_goal_distances_sq'], rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(data['candidate_goal_distances_sq'][:, -1], data['baseline_terminal_scores'],
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array([r['candidate_goal_distances_sq'] for r in rows]).T,
                                   data['candidate_goal_distances_sq'], rtol=1e-7, atol=1e-10)
        if any(r['active_goal_index'] != summary['active_goal_index'] or r['selected_candidate'] != selected for r in rows):
            raise ValueError(f'{folder.name}: goal or selected candidate changed during the fixed window')
        if [r['block'] for r in rows] != list(range(1, 17)):
            raise ValueError(f'{folder.name}: missing future block')
        if [r['control_step'] for r in rows] != [summary['anchor_control_step']+5*h for h in range(1, 17)]:
            raise ValueError(f'{folder.name}: time alignment mismatch')
        controls = [json.loads(line) for line in (folder/'task-steps.jsonl').read_text(encoding='utf-8').splitlines()]
        if [r['window_control_step'] for r in controls] != list(range(1, 81)):
            raise ValueError(f'{folder.name}: full 80 control actions were not executed')
        import imageio.v2 as imageio
        reader = imageio.get_reader(folder/'video.mp4')
        try:
            if reader.count_frames() != 17:
                raise ValueError(f'{folder.name}: expected initial plus 16 observed video frames')
        finally:
            reader.close()
        rank = 1+(data['candidate_goal_distances_sq'] < data['candidate_goal_distances_sq'][selected]).sum(axis=0)
        valid.append(dict(folder=folder, summary=summary, data=data, rows=rows, rank=rank))
    if len(all_cases) != manifest['total_source_cases']:
        raise ValueError('incomplete source cohort accounting')
    if not valid:
        raise ValueError('no valid first-box measurement windows')
    return manifest, all_cases, valid


def band(ax, times, values, label, color):
    q1, median, q3 = np.nanpercentile(values, [25, 50, 75], axis=0)
    ax.fill_between(times, q1, q3, alpha=.18, color=color, linewidth=0)
    ax.plot(times, median, '-o', ms=3, color=color, label=label)


def plot_global(cases, out):
    plt = plotting()
    times = np.arange(1, 17)*.5
    mse = np.stack([c['data']['latent_mse'] for c in cases])
    predicted = np.stack([c['data']['imagined_goal_distance_sq'] for c in cases])
    actual = np.stack([c['data']['actual_goal_distance_sq'] for c in cases])
    initial = np.array([c['summary']['initial_goal_distance_sq'] for c in cases])
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    band(ax, times, mse, 'Median; shaded IQR', '#2673a6')
    ax.plot(times, mse.mean(0), '--', color='#ce7e27', label='Mean')
    ax.set(xlabel='Open-loop execution time after anchor (s)', ylabel='Latent MSE (mean over dimensions)',
           title=f'One fixed WM rollout vs actual execution | {len(cases)} post-first-box windows')
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.legend()
    fig.savefig(out/'mse-vs-horizon.png')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, normalized in zip(axes, [False, True]):
        denom = np.where(initial > 0, initial, np.nan)[:, None] if normalized else 1.
        band(ax, times, predicted/denom, 'Imagined median; shaded IQR', '#2673a6')
        band(ax, times, actual/denom, 'Actual median; shaded IQR', '#ce7e27')
        ax.set(xlabel='Open-loop execution time (s)',
               ylabel='Distance / initial squared distance' if normalized else 'Squared latent distance to fixed subgoal',
               title='Normalized to anchor distance' if normalized else 'Same fixed active subgoal, every future latent')
        if normalized:
            ax.axhline(1, color='#777777', lw=1, ls='--')
        ax.legend(fontsize=8)
    fig.savefig(out/'goal-distance-vs-horizon.png')
    plt.close(fig)
    thresholds = np.array([c['summary']['goal_threshold_sq'] for c in cases])[:, None]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for distances, label, color in [(predicted, 'Imagined', '#2673a6'), (actual, 'Actual', '#ce7e27')]:
        met = distances < thresholds
        ax.plot(times, met.mean(0), '-o', ms=3, color=color, label=label+' meets threshold at this time')
        ax.plot(times, np.maximum.accumulate(met, axis=1).mean(0), '--', color=color,
                label=label+' has met threshold by this time')
    ax.set(xlabel='Open-loop execution time (s)', ylabel='Fraction of measurement windows', ylim=(0, 1.03),
           title='Fixed subgoal latent-distance criterion: transient vs endpoint attainment')
    ax.legend(fontsize=8)
    fig.savefig(out/'threshold-attainment.png')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(6, len(cases)*.19)), constrained_layout=True)
    labels = [f'{c["summary"]["scene_seed"]} / {c["summary"]["sample_seed"]}' for c in cases]
    im = axes[0].imshow(np.log10(np.maximum(mse, 1e-12)), cmap='viridis', aspect='auto',
                        extent=[.25, 8.25, len(cases)-.5, -.5])
    axes[0].set(xlabel='Open-loop time (s)', title='Actual vs imagined latent MSE',
                yticks=range(len(cases)), yticklabels=labels, ylabel='Scene / sampler seed')
    axes[0].tick_params(axis='y', labelsize=7)
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], label='log10 MSE')
    ranks = np.stack([c['rank'] for c in cases])
    im = axes[1].imshow(ranks, cmap='viridis', aspect='auto', vmin=1, vmax=16,
                        extent=[.25, 8.25, len(cases)-.5, -.5])
    axes[1].set(xlabel='Imagined time (s)', title='Terminal-selected candidate rank at each future time',
                yticks=range(len(cases)), yticklabels=[])
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], label='Rank among 16 candidates (1 = smallest goal distance)')
    fig.savefig(out/'mse-and-candidate-rank-heatmap.png')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    x, y = predicted[:, -1], actual[:, -1]
    colors = np.array([c['summary']['action_clipped_fraction'] for c in cases])
    im = ax.scatter(x, y, c=colors, cmap='viridis', s=35, alpha=.8)
    low = min(x.min(), y.min())
    high = max(x.max(), y.max())
    ax.plot([low, high], [low, high], '--', color='#777777', label='Imagined = actual')
    ax.set(xlabel='Imagined squared distance at 8 s', ylabel='Actual squared distance at 8 s',
           title='Fixed candidate, fixed subgoal, complete execution')
    ax.legend()
    fig.colorbar(im, ax=ax, label='Fraction of selected action coordinates clipped')
    fig.savefig(out/'predicted-vs-actual-endpoint.png')
    plt.close(fig)


def plot_case(case, out):
    plt = plotting()
    s, d, rows = case['summary'], case['data'], case['rows']
    t = np.arange(1, 17)*.5
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(t, d['latent_mse'], '-o', ms=3)
    axes[0, 0].set(xlabel='Open-loop time (s)', ylabel='Latent MSE', title='Imagined vs actual, same action sequence')
    axes[0, 0].set_yscale('symlog', linthresh=1e-6)
    axes[0, 1].plot(t, d['imagined_goal_distance_sq'], '-o', ms=3, label='Imagined')
    axes[0, 1].plot(t, d['actual_goal_distance_sq'], '-o', ms=3, color='#ce7e27', label='Actual')
    axes[0, 1].axhline(s['goal_threshold_sq'], color='#777777', ls='--', label='Switch threshold (goal frozen)')
    axes[0, 1].set(xlabel='Open-loop time (s)', ylabel='Squared distance to fixed subgoal',
                   title=f'Active goal {s["active_goal_index"]}; selected candidate {s["selected_candidate"]}')
    axes[0, 1].legend(fontsize=8)
    costs = d['candidate_goal_distances_sq']
    im = axes[1, 0].imshow(costs, cmap='viridis', aspect='auto', extent=[.25, 8.25, 15.5, -.5])
    axes[1, 0].axhline(s['selected_candidate'], color='#e9aa62', lw=1.5, label='Terminal-selected candidate')
    axes[1, 0].set(xlabel='Imagined time (s)', ylabel='Candidate index', yticks=range(16),
                   title='Every latent scored; selection remains terminal-only')
    axes[1, 0].grid(False)
    axes[1, 0].legend(fontsize=8)
    fig.colorbar(im, ax=axes[1, 0], label='Squared goal distance')
    axes[1, 1].step(t, case['rank'], where='mid', label='Selected candidate rank', color='#2673a6')
    axes[1, 1].set(xlabel='Time (s)', ylabel='Rank among imagined candidates', ylim=(.5, 16.5),
                   title=f'Anchor at {s["anchor_control_step"]*.1:.1f}s; clipped coordinates {s["action_clipped_fraction"]:.1%}')
    twin = axes[1, 1].twinx()
    twin.step(np.r_[0., t], [s['initial_count']]+[r['stacked_count'] for r in rows], where='post', color='#ce7e27')
    twin.set(ylabel='Actual counted boxes', ylim=(-.1, 4.1))
    fig.suptitle(f'{case["folder"].name} | exactly one WM rollout in this 8 s window')
    fig.savefig(out/f'{case["folder"].name}.png')
    plt.close(fig)


def build_report(root, skip_case_plots=False):
    root = Path(root).resolve()
    manifest, cohort, cases = load_and_verify(root)
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    plot_global(cases, out)
    for case in cases:
        if not skip_case_plots:
            plot_case(case, out)
        elif not (out/f'{case["folder"].name}.png').exists():
            raise ValueError('cannot skip case plots before the first complete report generation')
    mse = np.stack([c['data']['latent_mse'] for c in cases])
    pred = np.stack([c['data']['imagined_goal_distance_sq'] for c in cases])
    actual = np.stack([c['data']['actual_goal_distance_sq'] for c in cases])
    ranks = np.stack([c['rank'] for c in cases])
    thresholds = np.array([c['summary']['goal_threshold_sq'] for c in cases])[:, None]
    imagined_met, actual_met = pred < thresholds, actual < thresholds
    imagined_ever = np.maximum.accumulate(imagined_met, axis=1)
    actual_ever = np.maximum.accumulate(actual_met, axis=1)
    horizons = []
    for h in range(16):
        horizons.append(dict(block=h+1, seconds=(h+1)*.5, windows=len(cases),
                             mse_mean=float(mse[:, h].mean()), mse_std=float(mse[:, h].std()),
                             mse_median=float(np.median(mse[:, h])), mse_p25=float(np.percentile(mse[:, h], 25)),
                             mse_p75=float(np.percentile(mse[:, h], 75)), mse_p95=float(np.percentile(mse[:, h], 95)),
                             imagined_goal_distance_median=float(np.median(pred[:, h])),
                             actual_goal_distance_median=float(np.median(actual[:, h])),
                             selected_candidate_rank_median=float(np.median(ranks[:, h])),
                             selected_is_best_at_this_horizon_fraction=float(np.mean(ranks[:, h]==1)),
                             imagined_threshold_met_fraction=float(imagined_met[:, h].mean()),
                             actual_threshold_met_fraction=float(actual_met[:, h].mean()),
                             imagined_ever_met_fraction=float(imagined_ever[:, h].mean()),
                             actual_ever_met_fraction=float(actual_ever[:, h].mean())))
    write_json(out/'horizon-summary.json', horizons)
    with (out/'horizon-summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(horizons[0]))
        writer.writeheader()
        writer.writerows(horizons)
    summary = dict(source_cases=len(cohort), measured_windows=len(cases), excluded=len(cohort)-len(cases),
                   wm_rollouts_in_measurement=sum(c['summary']['model_calls']['test_rollout'] for c in cases),
                   prefix_wm_rollouts=sum(c.get('model_calls', {}).get('prefix_rollout', 0) for c in cohort),
                   mse_first_median=float(np.median(mse[:, 0])), mse_final_median=float(np.median(mse[:, -1])),
                   within_window_mse_growth_median=float(np.median(mse[:, -1]/np.maximum(mse[:, 0], 1e-12))),
                   actual_endpoint_distance_larger_fraction=float(np.mean(actual[:, -1]>pred[:, -1])),
                   imagined_endpoint_distance_median=float(np.median(pred[:, -1])),
                   actual_endpoint_distance_median=float(np.median(actual[:, -1])),
                   clipping_fraction_median=float(np.median([c['summary']['action_clipped_fraction'] for c in cases])))
    summary.update(imagined_earlier_distance_smaller_cases=int((pred[:, :-1].min(1)<pred[:, -1]).sum()),
                   actual_earlier_distance_smaller_cases=int((actual[:, :-1].min(1)<actual[:, -1]).sum()),
                   imagined_threshold_met_any_cases=int(imagined_ever[:, -1].sum()),
                   imagined_threshold_met_endpoint_cases=int(imagined_met[:, -1].sum()),
                   actual_threshold_met_any_cases=int(actual_ever[:, -1].sum()),
                   actual_threshold_met_endpoint_cases=int(actual_met[:, -1].sum()),
                   actual_mid_met_but_endpoint_not_cases=int((actual_ever[:, -1]&~actual_met[:, -1]).sum()))
    write_json(out/'aggregate.json', summary)
    write_json(out/'case-metrics.json', [c['summary'] for c in cases])
    write_json(out/'verification.json', dict(status='passed', measured_windows=len(cases),
                                            future_latent_pairs=16*len(cases), executed_control_steps=80*len(cases),
                                            videos_decoded=len(cases), candidate_count=16, wm_rollouts_per_window=1))
    selected_data = {c['folder'].name: c['summary'] for c in cases}
    excluded_rows = ''.join(f'<tr><td>{c["scene_seed"]}</td><td>{c["sample_seed"]}</td><td>{html.escape(c["reason"])}</td></tr>'
                            for c in cohort if c['status']!='complete')
    doc = '''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Post-first-box open-loop evidence</title><style>body{font:15px system-ui;color:#253644;max-width:1400px;margin:25px auto;padding:0 20px;background:#fafbfc}img{max-width:100%}select{font:inherit;padding:7px}.row{display:flex;gap:20px;align-items:flex-start}.row video{width:320px;max-width:35vw}.row img{max-width:72vw;width:950px}td,th{padding:6px;text-align:left}p{line-height:1.6}a{color:#176a9a}@media(max-width:700px){.row{display:block}.row video,.row img{max-width:100%;width:100%}}</style>
<h1>Post-first-box: one fixed 16-block rollout</h1><p>SUMMARY</p>
<p>Prefixes are regenerated B1 trajectories, not exact restorations of the original runs. Prefix WM calls are separate.
Each measurement window samples 16 candidates once, makes one batched WM rollout, selects by the last latent only,
then executes all 80 control actions without replanning. The active subgoal is fixed throughout the window.</p>
<select id="case">OPTIONS</select><p id="info"></p><div class="row"><video id="video" controls preload="none"></video><a id="figureLink"><img id="figure" alt="Case evidence"></a></div>
<p><a id="steps">16-block metrics JSONL</a> · <a id="arrays">Complete imagined/actual arrays NPZ</a> · <a id="anchor">Anchor state NPZ</a></p>
<h2>Imagined vs actual latent MSE</h2><img src="mse-vs-horizon.png" alt="MSE by horizon">
<h2>Every latent vs the same fixed subgoal</h2><img src="goal-distance-vs-horizon.png" alt="Imagined and actual goal distances">
<img src="threshold-attainment.png" alt="Transient and endpoint attainment of the fixed latent goal threshold">
<p><a href="mse-and-candidate-rank-heatmap.png">All-window MSE and candidate-rank heatmap</a> · <a href="horizon-summary.csv">Horizon summary CSV</a> · <a href="REPORT.md">Definitions and interpretation</a></p>
<img src="predicted-vs-actual-endpoint.png" alt="Imagined vs actual terminal goal distance">
<p>Actions use the baseline clipping convention. The WM receives sampled normalized actions; the simulator executes clipped raw actions.
This measures overall prediction-execution agreement, including clipping effects. No per-latent aggregate is used to change selection.</p>
<h2>Excluded source cases</h2><table><tr><th>Scene</th><th>Seed</th><th>Reason</th></tr>EXCLUDED</table>
<script>const cases=DATA;const select=document.getElementById('case');function update(){const name=select.value,m=cases[name],base='../cases/'+name+'/';document.getElementById('video').src=base+'video.mp4';document.getElementById('figure').src=name+'.png';document.getElementById('figureLink').href=name+'.png';document.getElementById('steps').href=base+'steps.jsonl';document.getElementById('arrays').href=base+'trajectories.npz';document.getElementById('anchor').href=base+'anchor-state.npz';document.getElementById('info').textContent='anchor='+m.anchor_control_step/10+'s; fixed goal='+m.active_goal_index+'; terminal-selected candidate='+m.selected_candidate+'; MSE(8s)='+m.mse_final.toPrecision(4);}select.addEventListener('change',update);update();</script></html>'''
    doc = doc.replace('SUMMARY', html.escape(f'{len(cohort)} source cases; {len(cases)} measured windows; {len(cohort)-len(cases)} excluded.'))
    doc = doc.replace('OPTIONS', ''.join(f'<option>{name}</option>' for name in selected_data))
    doc = doc.replace('EXCLUDED', excluded_rows).replace('DATA', json.dumps(selected_data, allow_nan=False))
    (out/'index.html').write_text(doc, encoding='utf-8')
    lines = ['# 第一箱后的单次 16-block rollout 诊断', '',
             f'原始 cohort 共 {len(cohort)} 例；有效测量窗口 {len(cases)}，排除 {len(cohort)-len(cases)}。', '',
             '实验口径：原始 run 没有可恢复的完整中途状态，因此按同 scene/seed 重新生成 B1 前缀。原始未达到第一箱的案例预先排除；其他案例取首次当前计数至少为 1 的 block 边界，并要求剩余时间够执行 80 control steps。', '',
             '前缀准备会多次调用 WM，不属于测量窗口。每个测量窗口仅采样一次候选、调用一次 batched WM rollout（16 candidates × 16 future blocks）。选择仍按最后一个 latent 到当前子目标的平方距离最小，不使用均值评分换候选。', '',
             '选中后完整执行 16 blocks / 8 秒，不重规划，也不切换 active subgoal。每执行完 5 个 control steps 渲染真实图像，最后统一编码为 16 个真实 future latents；起点 latent 单独保存，不与第一步预测错位。', '',
             '| 指标 | 值 |', '| --- | --- |',
             f'| 0.5s MSE 中位数 | {summary["mse_first_median"]:.6g} |',
             f'| 8s MSE 中位数 | {summary["mse_final_median"]:.6g} |',
             f'| 窗口内 MSE(8s)/MSE(0.5s) 比值的中位数 | {summary["within_window_mse_growth_median"]:.4g} |',
             f'| 8s 实际目标距离大于预测目标距离的比例 | {summary["actual_endpoint_distance_larger_fraction"]:.1%} |', '',
             f'实际轨迹在中途某步满足固定目标 latent 距离阈值的窗口为 {summary["actual_threshold_met_any_cases"]}/{len(cases)}；8 秒终点满足的为 {summary["actual_threshold_met_endpoint_cases"]}/{len(cases)}；其中 {summary["actual_mid_met_but_endpoint_not_cases"]} 例中途满足、终点已离开。', '',
             '这是固定 subgoal 的 latent 距离条件，不等同堆叠任务完成。说明终点值省略了时序信息，但本次没有改变候选选择，不能直接证明逐步聚合评分会改善闭环性能。', '',
             'MSE 对 latent 维度取均值；目标距离对 latent 维度取平方和。每个 horizon 的 n、mean、std、median、P25/P75/P95 在 horizon-summary.json/csv。', '',
             '保留 baseline 的裁剪方式：WM 预测使用采样的 normalized actions，真实执行使用还原并裁剪到 [-1,1] 的 raw actions。因此误差包含动作裁剪影响，不能单独归因于 world model。clip fraction 已逐 block 保存。', '',
             '所有候选逐 latent 的目标距离均被保存，候选 rank 只用于描述。终点选中的候选在中途不是距离最小的候选，并不自动意味着选择错误。', '',
             '本实验的起点是重新生成的真实状态，不是旧轨迹的逐 bit 恢复；GPU/dtype 在 run.json 中。8 秒窗口的堆叠计数不当作原闭环 benchmark 的成功率。', '',
             '每例保存：anchor-state.json/npz、prefix-steps.jsonl、trajectories.npz（全部候选动作与 imagined latent、17 个 observed latents）、steps.jsonl、task-steps.jsonl、17 帧视频/PNG、末尾接触和支撑状态。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(out/'provenance.json', dict(report=file_identity(__file__), run=file_identity(root/'run.json'),
                                          arrays=[file_identity(c['folder']/'trajectories.npz') for c in cases]))
    print(json.dumps(summary, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--skip-case-plots', action='store_true', help='Reuse already-rendered case figures when updating aggregate statistics')
    args = p.parse_args()
    build_report(args.run, args.skip_case_plots)


if __name__ == '__main__':
    main()
