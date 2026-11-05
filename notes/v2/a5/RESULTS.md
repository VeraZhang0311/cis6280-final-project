# A5 on-policy WM finetuning：本轮未取得收益

日期：2026-11-05。

**结论：本轮 on-policy WM finetuning 效果不好，未改善预设的 calibration 综合排序指标；用户指定的 56k WM 在 32-candidate 闭环测评中，平均任务表现也低于原 WM。当前保留原 WM 作为默认。** 这一结论限于本轮数据、配方和开发场景，不代表所有 on-policy finetuning 方法均无效。

## 配置与完成情况

- 固定 A4 的累计 120k actor；encoder/projector、normalizers 及其 buffers 保持不变，只训练 WM action_encoder、predictor、pred_proj 的可训练参数。
- 采集完成：5000 个 WM-train 锚点 × 16 候选，500 个 calibration 锚点 × 32 候选，每候选完整执行 16 blocks；每个 split 的 A4-16/A4-32 主轨迹行为各占一半。冻结检查通过，采集记录没有数值异常排除。
- 采集 Job `1502656`：完成，耗时 3:41:29；WM 训练 Job `1502663`：完成 60k steps，耗时 4:05:51；均在共享 A10 上运行。
- [W&B 6280_wm](https://forge.coreweave.com/wandb/tianzizh-university-of-pennsylvania/6280_wm/runs/a5-wm-onpolicy-120k-s0)。完整配方、模型身份和场景划分见 [launch.json](launch.json) 与 [collection-plan.json](collection-plan.json)。

## 预设选模：保留 step 0 原 WM

在固定的 500 个 calibration 锚点上，使用相同 32 条真实候选分支，分别评估前 16 条与全部 32 条。预测/真实候选 cost 都是 16 个未来 latent 对 active goal 的最小平方距离。按 `(regret_16 + regret_32) / 2` 选模，越低越好；step 0 参与，平分保留较早 checkpoint。

| WM checkpoint | h=16 latent MSE | regret_16 | regret_32 | 综合 regret |
| --- | ---: | ---: | ---: | ---: |
| 原 WM / step 0 | 0.004388 | 0.44253 | 0.56587 | **0.50420** |
| 56k：最佳微调 checkpoint | 0.004052 | 0.45690 | 0.55758 | 0.50724 |
| 60k：训练末步 | 0.004054 | 0.48371 | 0.54786 | 0.51578 |

step 0 与每 2000 步共检查 31 个 checkpoints，所有微调 checkpoint 的综合 regret 均未优于原 WM，因此正式选模为 **step 0**（`original_retained=true`）。56k 的 h=16 MSE 更低、regret_32 略低，但 regret_16 更高，综合排序指标未改善。预测 MSE 降低没有转化为本轮的选模收益。

证据：[全部 calibration 指标](evidence/calibration-metrics.json)、[原始选模结果](evidence/wm-selection.json)。

## 用户指定 56k WM 的 32-candidate 补充测评

用户在得知原选模结果后指定额外测试 56k WM。它不是预设规则选出的 WM；原 `selection.json` 与权重 metadata 均未修改。独立补充实验固定 actor=120k、WM=56k、32 candidates，复用相同 20 个开发 scenes × seeds 0/2/3，共 60 条；与原 WM 的 A4-32 在相同 L40、bfloat16、8 workers、goals 和评分/执行配置下配对比较。

| 指标 | A4-32：原 WM | A5-32：56k WM | A5 − A4 |
| --- | ---: | ---: | ---: |
| mean peak boxes | 2.8667 | 2.5833 | −0.2833 |
| mean final boxes | 2.1333 | 1.8333 | −0.3000 |
| success | 14/60，23.33% | 11/60，18.33% | −5.00 个百分点 |
| peak ≥3 箱 | 41/60，68.33% | 34/60，56.67% | −11.67 个百分点 |
| 平均决策耗时 | 217.01 ms | 216.51 ms | −0.50 ms |
| P95 决策耗时 | 247.56 ms | 247.48 ms | −0.08 ms |

Job `1504384` 正常完成 60/60 条，耗时 12:22；60 个视频、`steps.jsonl`、`counts.jsonl` 与 episode 统计均保存。没有观察到在线耗时方面的实质变化。

按 scene 分组、2000 次配对 bootstrap 得到的 A5−A4 95% 区间：peak boxes `[−0.6167, 0.0000]`，final boxes `[−0.6833, 0.1000]`，success `[−20, +10]` 个百分点。区间均包含或触及 0，因此记录为**平均表现下降、未看到闭环收益**，不将结果表述为已证明稳定或普遍退步。

证据：[完整结果与配对区间](evidence/wm56k32-summary.json)、[运行 manifest](evidence/wm56k32-run.json)、[56k checkpoint 身份及显式请求记录](evidence/wm56k32-checkpoint-receipt.json)。

## 证据路径与范围

```text
cluster A5 采集/训练工作区：
  /mnt/projects/ll/tianzizh/cis6280-v2-a5-20261105
WM checkpoints / selection / train log：
  checkpoints/wm-a5/
cluster 56k 补充测评工作区：
  /mnt/projects/ll/tianzizh/cis6280-v2-a5-eval56k-20261105
测评结果与视频：
  outputs/planning/a5-56k-32/A5-32/episodes/scene-<scene>.sample-<seed>/
```

本轮实际额外闭环测试只有 **56k / A5-32**，没有运行微调 WM 的 A5-16，也没有运行预留正式验证 scenes。保留原 WM 的预设选模结果；后续报告将本轮 WM 微调作为未取得收益的尝试，并将 56k 补充实验与预设选模结果分开报告。现有证据不能单独定位原因，不能据此认定是过拟合、数据不足或某一种特定分布变化造成。
