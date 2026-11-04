# Planning track：第一阶段代码与运行

本项目用冻结的官方 policy / world model，从 front、side 两路腕部相机图像规划四箱堆叠。第一阶段按 [DESIGN.md](DESIGN.md) 诊断 B0/B1，不改采样、目标切换、评分或控制逻辑。

后续实验范围已按用户要求改为 **只关注 B1（with planning）**。runner 和 cluster 提交脚本默认只运行 B1，分析与案例标注也以 B1 为主；历史 B0 结果保留。只有显式传入 `--baselines B0 B1` 才运行两组对照。

此默认值已更新到本地代码与提交脚本，并在 SSH 恢复后同步到 cluster。本批任务显式使用 `--baselines B1`。

## B1：20 场景批量诊断

SSH 已恢复，本地 B1 默认入口与终止帧修复已同步。固定场景清单为 [validation-scenes-20.json](validation-scenes-20.json)：提供的 val100 goals 按 scene 编号取前 20 个，包含 100038，不按结果筛选。每个 scene 跑 sampler seeds 0、2、3，共 60 episodes；全部保存视频。

本批 Slurm job 为 `1497072`，使用 `liu-compute` 的一张 L40、4 workers，保留官方 B1 参数。提交脚本为 `scripts/phase1_b1_20.sbatch`，完整结果目录为 `outputs/diagnostics/b1-20-1497072/`。

每个 episode 保存 `steps.jsonl`、`events.jsonl`、`summary.json`、`timeline.png`、包含终止帧的 `video.mp4`；另保存末尾 56 个 control steps 的 `tail-state.json`、物体索引/颜色/尺寸映射 `objects.json`，以及末尾六张标注关键帧 `keyframes/step-*.png`。末尾状态直接从 evaluator 读取，不进入 planner。新批次没有对应的原始轨迹用于复现比较，因此不生成含糊的 `reproduction.json`。

离线分析（仅需 CPU）：

```bash
python -m assignment.training.planning.report outputs/diagnostics/b1-20-1497072 --verify-videos
```

结果写到 `analysis/`：`index.html` 可选择 scene/seed 同看视频和六联诊断图；`outcomes.png`、`task-progress.png` 汇总任务进展；`steps-aggregate.png`、两张 heatmap 展示全批次 block 信号；`after-first-box.png` 与 `first-to-second.png` 展示第一箱前后的变化及截尾观测；每个 episode 另有 `detail-scene-*.sample-*.png`。派生数值在 `episode-metrics.json/csv`，输入哈希在 `provenance.json`，跨文件一致性检查在 `verification.json`。

### 本批结果与可观察证据

任务 `1497072` 正常完成，exit 0，耗时 22 分 33 秒。60 个视频、360 张末尾关键帧、21,540 条 block 记录、344,640 个候选评分、3,360 条末尾逐 control-step 状态均已下载和检查。9 项诊断测试与 4 项离线分析测试通过，报告的 60 个 episode 选择入口也已核对。批量任务已释放 GPU。

| 指标 | B1 / 60 episodes |
| --- | --- |
| 曾达到至少 1 箱 | 47 / 60（78.3%） |
| 曾达到至少 2 箱 | 25 / 60（41.7%） |
| 曾达到至少 3 箱 | 8 / 60（13.3%） |
| 曾达到 4 箱 / 四箱成功 | 0 / 60 |
| 平均 peak / final | 1.333 / 1.083 |
| 达到第一箱后未到第二箱 | 22 / 47 |
| 曾在某个非末尾子目标停留超过 60 秒 | 51 / 60 |
| 一次跳过多个子目标 | 0 / 60 |

全批次候选最佳与次佳的相对分差中位数为 8.88%，约 10.08% 的 block 分差低于 1%；因此不能概括为所有候选评分都相同，但这也不证明有效候选覆盖充分。一个重要对照是 scene 100069 / seed 0：非末尾目标停留达到 157.5 秒，仍堆到 3 箱，说明目标滞留不能直接等同任务停滞。图表用于定位案例和提出假设，不据这些相关信号单独认定失败原因。

本地与 cluster 均保留完整结果目录；报告入口为 `outputs/diagnostics/b1-20-1497072/analysis/index.html`，图表定义和解释限制在同目录的 `REPORT.md`。

## 运行

使用官方三个文件，显式传 world model，不修改 checkpoint 内的 encoder 路径。准备依赖与手模型：`bash scripts/setup.sh`。仅需模型、goals，不需要下载训练集。

```bash
hf download YongYong/cis6280-stack-world-models-v2 policy/df-smwm-c1-sigma04/policy.pt jepa/smwm-c3/model.pt --local-dir checkpoints
hf download YongYong/cis6280-stack-datasets-v2 --repo-type dataset goals/scripted-val100-every80/goals.npz --local-dir dataset
export MUJOCO_GL=egl
python -m assignment.training.planning.phase1 \
  --policy checkpoints/policy/df-smwm-c1-sigma04/policy.pt \
  --world-model checkpoints/jepa/smwm-c3/model.pt \
  --goals dataset/goals/scripted-val100-every80/goals.npz \
  --scene-seeds 100038 --sample-seeds 0 2 3 --workers 2 --videos 1 \
  --out outputs/diagnostics/scene100038-v1
```

入口默认运行固定参数的 B1、128px 相机、4 箱、1795 steps、8 秒子目标；官方 goals 最后的成功帧可以不落在 80-step 网格上。每个 episode 使用 `scene_seed + 1_000_000 * sample_seed`，与原 evaluator 一致。输出目录必须是新的，避免覆盖。

随后用 `--scene-file notes/validation-scenes.json` 指定固定 JSON 整数数组，先扩到 10–20 个场景；根据首次结果选取不同进展的案例，再决定固定 50 scenes。不根据未经验证的表现预先标注场景。`--videos N` 对前 N 个场景的所有 sampler seeds、两个 baseline 都保存视频。

Cluster 登录：`ssh tianzizh@grasp-login2.seas.upenn.edu`。项目存储使用 `/mnt/projects/ll/tianzizh` 下的独立目录，优先申请 `liu-compute`（账号 `liu-account`，QOS `ll-med`）；无空闲再使用共享 `batch`，QOS 以账号实际权限为准。从项目根目录提交 `scripts/phase1_cluster.sbatch`，通过环境变量 `PHASE1_PYTHON` 指定兼容的已有 Python（只复用依赖，不修改其他项目的环境）。提前创建 `outputs/`，供 Slurm 写日志。缺少的视频编码包可以用 `$PHASE1_PYTHON -m pip install --target outputs/runtime-deps --no-deps imageio-ffmpeg` 独立补充，脚本会自动加入该依赖路径。`--gpus 0` 表示 Slurm 分配中的第一个 GPU，worker 会保留分配边界。

## 输出与解释

```text
<run>/run.json                   # 完整场景/seed、固定参数、输入 SHA256、代码哈希、GPU/dtype
<run>/summary.json               # B0/B1 无条件成功率、均值、条件转化率、决策耗时
<run>/evidence.json              # 可观察的诊断摘要，供选取案例，不推断失败原因
<run>/review.csv                 # 视频人工标注：接近/接触/抬起另一箱、静止、破坏堆叠、假设
<run>/B0/episodes/scene-100038.sample-0/
  steps.jsonl
  events.jsonl
  summary.json
  video.mp4
  timeline.png
<run>/B1/episodes/...
```

每条 step 以动作执行前真实观测的时间为准（0、5、10…），当前 stack 是该时刻已有任务信号。任务事件在每个 control step 后记录（1、2、3…），因此不会漏掉采样点之间的首次放置或计数下降。第一次选中目标也写 `goal_changed`（previous_index=null），此后保留跳转前后索引。目标停留时间从首次选中/切换时刻计算。

`plan_change_rms` 比较上一轮选中计划的第二个 block 与现在执行的第一个 block，使用实际 raw/clipped action 单位。`prediction_mse` 比较上一轮选中 rollout 的第一步 latent 与现在真实编码；rollout 使用 baseline 本来的候选动作，含动作裁剪带来的差异，所以只解释为整体预测—执行一致性。首条均为 null；B0 不增加 predictor 调用，目标和规划评分字段均为 null。诊断入口固定 execute=1，不将该对齐方式套用于多 block 执行。

未到第二箱的 `first_to_second_s=null`，单独记录 `post_first_observed_s`，不把截尾观测当成功耗时。汇总 mean / P95 使用所有 block 的总决策耗时，包含诊断提取，排除渲染、存盘、绘图；不平均 episode 的 P95。

### stack count 的具体口径

这里沿用 `StackTaskEnv.step()` 的 `tower()` 瞬时进度计数，不是从 RGB 数层数，也不是 `supported_height()` 的三秒稳定性评分。`structure()` 先排除与手或地面接触的箱子；桌面根箱的整个 XY 包围盒须在目标区域内，且与桌面存在向上的承重支撑。再沿下箱 → 上箱的支撑边加入无手/地面接触的箱子。`tower()` 只取其中箱数最多的连通结构，分开的箱子不相加，箱数相同才按高度选。

支撑边的判定是接触压缩力 > 0.0001 N、向上法向分量 > 0.5，且上下箱中心高度顺序正确。手接触排除使用 `object_hand_contacts != 0`，不是 grasp 标签的 0.05 N 阈值；轻触也可能影响瞬时计数。上箱不另做目标区域包围盒检查，目标区域条件用于桌面根箱。

`max_placed` 是整个 episode 的最大计数，`final_count` 仅取最后一个 control step。因此接触或支撑关系在最后时刻变化，即使肉眼外观近似不变，final 也可能低于 peak。四箱 `success` 还要求满足目标且完全脱手连续 10 control steps。

初版六个视频只渲染动作 block 开始时刻，最后一帧为 179.0s，而 summary 取 179.5s 的终止状态。现已修正 evaluator：诊断视频追加带 `final` 标记的终止帧，不额外调用 agent，也不改变动作或评分。原始 run 视频保留；复核视频另存，避免混淆原始证据。

离线重绘与汇总（无需 GPU 或 checkpoint）：

```bash
python -m assignment.training.planning.diagnostics outputs/diagnostics/scene100038-v1
python -m unittest discover -s tests -p test_planning_diagnostics.py -v
```

离线脚本保留已有 `review.csv` 手工注释。结合视频填写 observed_behavior，再写 hypothesis / next_check；候选评分接近不能单独证明候选覆盖不足，计数下降也不能直接判为倒塌。第一阶段结论需要真实运行与人工视频检查，代码测试不能代替这些证据。

## 首批端到端验证：2026-11-03

Slurm job `1496824` 在共享 `batch` 分区完成（exit 0，3 分 31 秒）；此前 `liu-compute` 的限时申请因资源忙未能分配。没有更改原有训练任务。

场景 `100038`，两个 baseline 使用相同的官方输入文件与 sampler seeds：

| Sampler seed | B0 peak / final | B1 peak / final |
| --- | --- | --- |
| 0 | 1 / 1 | 3 / 2 |
| 2 | 0 / 0 | 1 / 0 |
| 3 | 2 / 2 | 2 / 1 |

所有 episode 均完成 1795 control steps，没有达到四箱成功条件。平均 peak：B0=1，B1=2；这只是单场景 smoke test，不能作为 planner 改进的总体结论。B1 seed 2 的目标索引最终为 2，单个目标停留最长 120 秒；该记录可用于定位视频，尚不能证明停滞原因。计数回落需要视频确认，不能直接称为倒塌。

本次复用 cluster 的现有兼容环境，版本与仓库 lockfile 不完全相同：NVIDIA A10、PyTorch 2.11.0+cu128、MuJoCo 3.15.0、NumPy 2.5.1；encoder / predictor 使用 bfloat16 autocast，模型参数和 policy 推理为 float32。完整版本、SHA256、真实代码快照标识均在 run.json 中。后续对照必须固定同一套运行环境。

Cluster 原始结果：`/mnt/projects/ll/tianzizh/cis6280-final-project/outputs/diagnostics/scene100038-1496824/`。本地结果同步到 `outputs/diagnostics/scene100038-1496824/`。每个 episode 的视频、日志与时间线均已生成；视频的相机、目标和进度标注已抽帧检查。尚未完成全段行为人工标注，也尚未扩展 10–20 / 50 个 validation scenes，因此不在这里选择下一阶段算法方向。
