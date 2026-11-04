# 第一箱后的单次 WM rollout 实验

用户确认的实验定义：从第一箱放好后的一个真实状态出发，只规划一次；选中后完整执行 16 blocks / 8 秒，不重规划。仍按最后一个 imagined latent 到 active subgoal 的平方距离选择候选。记录每个 imagined latent 的目标距离，只作分析，不用均值等新评分改变选择。

原始 cohort：`outputs/diagnostics/b1-20-1497072/` 的 20 scenes × sampler seeds 0、2、3，共 60 例。其中 47 例曾达到第一箱；13 例原始未达到第一箱，预先标记不符合取样条件，不补新 scene。

原始 run 没有保存可恢复的完整中途状态或完整动作，因此不能从旧视频/标量日志精确恢复其第一箱时刻。本实验对原始符合资格的案例重新运行冻结的 B1 前缀，取第一次当前 stack count >= 1 的 block 边界，且剩余原任务时间必须够执行完整 80 control steps。未找到合格起点的案例另报原因。

**WM 只 rollout 一次指测量窗口。** 起点准备沿用原 B1，会多次调用 WM，单独记作 `prefix_rollout`。窗口中只采样一次 16 candidates，做一次 batched 16-block rollout，按终点评分选 best，然后连续执行该候选的全部动作。真实帧编码不调用 predictor。方法包装器禁止窗口内再次 rollout 或采样。

本次 Liu GPU 无空闲，预计需等待数小时；按用户给定优先级改用共享 2080 Ti，官方推理 dtype 为 float16。任务编号为 `1497731`。之前排队的本会话任务 `1497723` 已取消，原有训练任务未改变。此结果不是旧 L40 / bfloat16 轨迹的逐 bit 复现。

## 保存内容

结果目录：`outputs/openloop/postfirst-1497731/`。

- `run.json`：原 cohort 标识/输入 SHA256、真实配置、GPU/dtype、窗口定义。
- `cases/scene-*.sample-*/summary.json`：有效/排除状态、原始与重新生成的首次放置时间、anchor、所选候选、窗口模型调用次数。
- `anchor-state.json/npz`：真实 MuJoCo integration state、控制器目标/时间/状态、历史状态、agent context、sampler RNG state、实际 prefix actions，便于后续恢复分支。
- `trajectories.npz`：16 个候选的完整 normalized/raw actions、16×16 imagined latents、17 个 observed latents、固定目标 latent、全部逐步目标距离及 MSE。
- `steps.jsonl`：每个未来 block 一条，共 16 条；0.5 到 8 秒的 MSE、imagined/actual 目标距离、全部候选距离、裁剪比例与任务进度。
- `task-steps.jsonl`：80 个实际控制步骤的任务记录。
- `video.mp4`、`frames/`：起点及完整执行后 16 个实际观测，共 17 帧。
- `tail-state.json`、`objects.json`：末尾接触、支撑、逐箱状态与索引映射。

## 运行与可视化

```bash
python -m assignment.training.planning.openloop \
  --source-run outputs/diagnostics/b1-20-1497072 \
  --out outputs/openloop/new-postfirst-run --workers 4 --gpus 0

python -m assignment.training.planning.openloop_report outputs/openloop/postfirst-1497731
```

报告在 `analysis/index.html`，可按 case 选择视频与证据图。`mse-vs-horizon.png` 展示完整 16-step 预测误差；`goal-distance-vs-horizon.png` 比较 imagined/actual 对同一固定目标的距离；每例图中还有全部候选的逐 latent 距离热图。逐 horizon 的 n、mean、std、median、P25/P75/P95 保存为 JSON/CSV。

MSE 对 latent 维度求均值，目标距离对 latent 维度求平方和。保留 baseline 的动作裁剪方式：WM rollout 输入 sampled normalized actions，真实执行还原并裁剪的 raw actions；误差因此包含裁剪影响。每条实际轨迹确实执行同一固定候选，消除了中途重规划导致的动作错位。

## 完成结果

任务 `1497731` 正常完成，exit 0，耗时 5 分 29 秒。原始 60 例中，13 例原始未达到第一箱，另有 4 例重新生成前缀后未达到第一箱：100078/seed3、100087/seed0、100098/seed3、100152/seed0。因此有效窗口为 43，不补换案例。

窗口内共 43 次 WM rollout；起点准备另有 4,463 次 prefix rollout，不计入窗口测量。43 个完整视频、688 对 future latents、3,440 个实际 control steps、731 张真实帧、每例 16 个候选的完整预测/动作数组均已保存、下载并交叉核对；5 项相关测试通过，报告的 43 个 case 选择入口已核对。

| 指标 | 43 个有效窗口 |
| --- | --- |
| 0.5 秒 latent MSE 中位数 | 0.00122906 |
| 8 秒 latent MSE 中位数 | 0.00354368 |
| 同一窗口 MSE(8s)/MSE(0.5s) 比值的中位数 | 2.9013 |
| 8 秒实际目标距离大于预测目标距离 | 29 / 43（67.4%） |
| 8 秒 predicted / actual 目标平方距离中位数 | 18.982 / 19.894 |
| 实际中途满足固定目标 latent 阈值 | 34 / 43 |
| 实际终点满足固定目标 latent 阈值 | 20 / 43 |
| 实际中途满足、终点已离开 | 14 / 43 |

MSE 不是随 horizon 严格单调上升。选中候选的 predicted/actual 目标距离也可能先接近后远离当前目标；终点评分不会保留这段时序信息。但这里的阈值满足只是 latent 条件，不等同堆叠任务完成，也不证明改用逐步聚合评分会改善闭环性能。

只有 2 个窗口出现动作裁剪，各窗口裁剪比例最高为 0.05%，其余为零；原始比例保存在逐 block 日志。因本批 GPU/dtype 与旧 L40 不同、起点为重新生成，不将两批当作精确复现比较。

完整报告入口：`outputs/openloop/postfirst-1497731/analysis/index.html`。两类主要证据为 `mse-vs-horizon.png` 和 `goal-distance-vs-horizon.png`；补充的 `threshold-attainment.png` 展示瞬时与累计阈值满足比例。完整数字在 `horizon-summary.json/csv`。
