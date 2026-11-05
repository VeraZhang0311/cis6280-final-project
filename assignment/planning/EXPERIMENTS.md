# Planning 实验方案与运行记录

## 当前状态

截至 2026-11-05，B200 smoke 作业 `10085158`、筛选数组 `10085491` 的全部
24 个任务，以及汇总作业 `10085730` 均已完成。筛选共 720 episodes，观测最佳方法为
`trajectory`，剩余 270 个场景的确认实验尚未提交。
完整结果和运行记录见 [RESULTS.md](RESULTS.md)，可复算报告已提交到
[`results/planning/2026-11-05-screen/`](../../results/planning/2026-11-05-screen/README.md)。
受控 latent 测试只验证决策逻辑和接口，不能作为任务性能证据。

大文件路径：

- 模型：`/vast/projects/xuhw/healthworldx/zhangrm/model/cis6280-stack-world-models-v2`
- 子目标：`/vast/projects/xuhw/healthworldx/zhangrm/data/cis6280-stack-datasets-v2`
- 可通过 `PLANNING_MODEL_ROOT`、`PLANNING_DATA_ROOT` 或 runner 的参数覆盖。
- 项目 `.venv` 固定 `torch==2.9.1`、`mujoco==3.13.0`，实际依赖版本写入 environment.txt。

独立项目环境下 19 项测试通过，原始环境记录和测试日志随结果快照提交。

## 研究问题与方法

冻结官方视觉编码器、世界模型和 diffusion-forcing policy。
输入严格限制为当前/历史双视角 RGB、给定子目标图像和已执行动作；
不读取物体位姿、接触或环境奖励。环境状态仅由官方 evaluator 计算指标。

假设一：仅看预测终点可能错过中途已接近目标的候选。
设候选在第 t 个 block 的归一化距离为 d(t)，归一化尺度为当前相邻子目标的
latent 平方距离，使用

```text
arrival = argmin_t [d(t) + 0.05 * t/H]
cost = d(arrival) + 0.05 * arrival/H + 0.25 * d(H)
```

假设二：有序前瞻可以区分“到达后继续完成任务”和“到达后偏离”的动作。
只有 d(arrival) < tau 且存在严格晚于 arrival 的预测状态，才启用下个子目标奖励：

```text
cost -= 0.15 / (1 + min_{t > arrival} next_goal_distance(t))
```

该奖励有界，避免跳过当前目标直接追逐后续目标。保持官方 tau=0.25。

假设三：视觉状态可能已经越过当前子目标，但未触发原始半径条件。
连续 12 个 block 未取得至少 2% 的距离改善后，仅当下个子目标距离小于当前的
80%，且当前距离不超过相邻目标间距离的 1.5 倍，才允许前进一个目标。
不因超时而无条件跳过。恢复条件只使用真实观测 latent。

所有新规划器都先在原始动作空间裁剪到 [-1,1]，再归一化并交给世界模型，
使想象的动作与执行动作一致。官方 `goal` 默认评分和采样行为保持原样。

## 实验矩阵

| 名称 | 规划器 | 候选数 | 轨迹评分 | 有序前瞻 | 切换恢复 |
| --- | --- | --- | --- | --- | --- |
| policy | 官方无目标策略 | 1 | 否 | 否 | 否 |
| baseline | 官方 GoalPlanningAgent | 16 | 否 | 否 | 否 |
| clipbaseline | 官方评分，仅裁剪预测动作 | 16 | 否 | 否 | 否 |
| trajectory | 新规划器 | 16 | 是 | 否 | 否 |
| lookahead | 新规划器 | 16 | 是 | 是 | 否 |
| adaptive | 新规划器 | 16 | 是 | 否 | 是 |
| combined | 新规划器 | 16 | 是 | 是 | 是 |
| budget32 | 官方 GoalPlanningAgent | 32 | 否 | 否 | 否 |

新规划器各变体共享动作裁剪一致性处理。clipbaseline 单独测量动作裁剪的影响，
以区分评分收益与模型/执行一致性收益。
主要方法与 baseline 具有相同模型调用次数：16 候选、16-block 预测、
10 个 diffusion 求解步、执行 1 block 后重规划。budget32 作为额外计算预算对照。
实际耗时仍需测量，不能仅按调用次数宣称计算成本相同。

## 数据划分与统计

- smoke：前 2 个验证场景，策略种子 0，完整 1795 控制步，并保存首场景视频。
- screen：前 30 个验证场景，策略种子 0、2、3，8 个变体共 720 episodes。
- confirm：剩余 270 个场景，策略种子 0、2、3，baseline 与预先选定的最佳方法，1620 episodes。
- full：全部 300 个场景，每种方法 900 episodes；用于与作业表格对照。

scene_split 完全复用官方 SHA-256 分桶规则，300 场景列表写入 manifest。
screen 与 confirm 不重叠；confirm 也属于公开验证集，不称为隐藏测试集。
筛选后固定方法和超参数，再看 confirm。full 包含调参场景，需明确这一点。

主指标为 episode 内峰值堆叠数量的平均值；同时报告 >=1/2/3/4 比例、
ground fall 比例和每个 block 的 agent 耗时。所有方法配对相同 scene/sample seed。
统计工具先对每个 scene 的三个策略种子差值取均值，再按 scene 整组 bootstrap
10,000 次，报告差值的 95% 区间，避免把同场景三个随机种子视为独立场景。
确认集区间下界 >0 且没有明显耗时或失败率恶化，才支持方法有效的结论。

## PARCC 执行

已验证 `dgx-b200` 分区和 `dgx` QoS。PARCC 提交过滤器要求每张 B200
配套 28 CPU、224 GB 内存，不能使用原先设想的 8 CPU、48 GB。

```bash
cd /vast/home/z/zhangrm/wm/cis6280-final-project
sinfo -o '%P %a %l %D %G'
bash scripts/setup_planning.sh
mkdir -p runs/planning

sbatch --parsable --account=xuhw-healthworldx --partition=dgx-b200 --qos=dgx --gres=gpu:1 \
  --cpus-per-task=28 --mem=224G --time=01:00:00 \
  --output='runs/planning/%x-%j.log' \
  scripts/planning.slurm smoke baseline combined

# 确认 smoke 的视频、EGL/CUDA、动作范围和 JSON 结果后，再提交筛选。
# 每个任务一个变体、10 个场景、三个策略种子；最多同时占用两张 GPU。
sbatch --parsable --account=xuhw-healthworldx --partition=dgx-b200 --qos=dgx --gres=gpu:1 \
  --cpus-per-task=28 --mem=224G --time=00:30:00 --array=0-23%2 \
  --output='runs/planning/screen-%A_%a.log' \
  scripts/planning.slurm screen baseline combined policy clipbaseline trajectory lookahead adaptive budget32

# 待运行：固定筛选选中的 trajectory，确认集为 27 shards * 2 variants。
sbatch --parsable --account=xuhw-healthworldx --partition=dgx-b200 --qos=dgx --gres=gpu:1 \
  --cpus-per-task=28 --mem=224G --time=00:30:00 --array=0-53%2 \
  --output='runs/planning/confirm-%A_%a.log' \
  scripts/planning.slurm confirm baseline trajectory

.venv/bin/python -m assignment.planning.report \
  --baseline runs/planning/confirm/baseline/*.json \
  --candidate runs/planning/confirm/trajectory/*.json \
  --out runs/planning/confirm-comparison.json
```

筛选作业每个分片均在 10 分钟内完成，确认集使用同样大小分片，设置 30 分钟时限。
脚本默认 4 workers，单 GPU，各 worker 加载相同的冻结模型权重。
evaluator 每个 sample seed 写一个报告；单个报告内暂不支持 episode 级断点恢复，
中断时可按原数组索引重跑整个 shard。逐场景日志会保留中断前已完成的记录。
不要混合不同配置的报告目录。
安装脚本记录实际依赖版本与 checkpoint SHA-256，manifest 记录代码 SHA-256。

仅准备命令与 manifest，不占用 GPU：

```bash
python -m assignment.planning.experiments --stage smoke
python -m assignment.planning.experiments --stage screen --task-index 0 \
  --variants baseline combined policy clipbaseline trajectory lookahead adaptive budget32
```
