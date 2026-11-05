# v2.0：Goal-conditioned flow actor 与 on-policy WM finetuning

日期：2026-11-05。状态：累计 100k actor 的 A1/A2/A3 共 180 条，以及累计 **120k** actor 的 A4-16/A4-32 各 60 条开发测评均已完成。A5 on-policy 采集和 60k-step WM 微调已完成；预设 calibration 选模保留 **step 0 原 WM**。用户额外指定的 **56k WM / A5-32** 已完成 60 条，平均任务表现低于 A4-32，本轮未看到 WM 微调收益。完整结果与限制见 [A5 实验结论](a5/RESULTS.md)。

A5 采集 Job `1502656` 和 WM 训练 Job `1502663` 均已正常完成：5000/500 个 train/calibration 锚点、60k steps，分别耗时 3:41:29、4:05:51。训练 batch=256、microbatch=256，注册 W&B `6280_wm`。[原启动记录](a5/launch.json) 保存当时配置和依赖；[结果记录](a5/RESULTS.md) 保存最终状态、选模、闭环测评及证据。

本次执行变更覆盖下文原定 150k final EMA 的选择：保留原 50k 训练预算和调度作为历史配置，当前测评显式传入 `--a4-cumulative-step 120000`，使用独立 `a4-120k` 证据目录，并记录 checkpoint SHA256 与 `user_selected_intermediate_ema`。不将中间 EMA 改写为 150k final，不依据本次任务结果挑选 checkpoint。A4-16 相对 A2 的新增训练量为 **20k**。

120k actor 测评 Job `1502593` 已正常完成，使用共享 `batch` 的单张 L40、8 workers。A4-16/A4-32 各 60 条、共 120 条，均保存视频与逐步证据。[启动记录](../../outputs/planning/v2/a4-120k-20261104/launch.json)、[checkpoint 核验](../../outputs/planning/v2/a4-120k-20261104/preflight.json)。cluster 工作区为 `/mnt/projects/ll/tianzizh/cis6280-v2-a4-eval120k-20261104`，结果目录为其下 `outputs/planning/v2/a4-120k`。

本分支 `tianzi-branch/v2.0` 从 `main` 的 `9540172640099ac764e3eeca52d09265dbb018fb` 创建，不继承 v1.1 的 planner 实现。v1.1 已保存并推送至 `8c7ffdf1de2370e0fcca2a227f80663a1a3929c2`，其 [设计与结果记录](https://github.com/VeraZhang0311/cis6280-final-project/tree/8c7ffdf1de2370e0fcca2a227f80663a1a3929c2/notes/v1.1) 保持独立。

## 1. 目标与固定范围

原 actor 只根据当前 latent 生成动作，goal 在后续 planner 评分中才起作用。本轮直接训练一个接收 active goal 的 actor，验证目标条件能否改善动作提案，并比较直接执行、当前目标筛选和下一目标 lookahead。

这是 training track 的实验：A1–A4 训练 actor，encoder/projector、WM action encoder 和 predictor 全部冻结；A5 固定 A4 actor 与 encoder/projector，只微调 WM 的动作编码与预测部分。正式任务评估沿用官方每 8 秒的 subgoal 图片，不替换评估参考序列，不修改环境循环、任务定义或动作接口。

| 组件 | 本轮处理 |
| --- | --- |
| 图像 encoder 与 projector | 冻结，编码当前观测与 goal；可复用一致的 latent cache |
| WM action encoder / predictor / pred_proj | A1–A4 冻结；A5 采集时固定原 WM，训练时微调这些模块，评估时再次冻结 |
| Diffusion Forcing / rectified-flow actor | 初始阶段从官方 checkpoint 初始化；A4 从累计 100k goal actor 续训；A5 完整冻结与 A4 相同的 actor |
| 输出长度 | 固定 16 blocks |
| 每个 block | 5 control steps × 25 维，0.5 秒、125 维 |
| 执行与重规划 | 每轮只执行首 block，再读取真实观测 |
| actor 采样 | n_steps=10、lag=0 |
| 实际目标切换 | 官方真实观测距离规则，tau=0.25 |

本轮不加入局部 inverse 分支、gradient guidance、planner 候选的跨轮 warm start、CEM/MPPI、G6 补采样或停滞重对齐。A1/A2/A3 共用累计 100k 的 actor checkpoint；A4-16/A4-32 与 A5-16/A5-32 共用用户指定的尾部覆盖续训累计 120k EMA checkpoint。A5 在 A4 所用 WM 上做一次 on-policy finetuning，不重新训练 actor 或图像表示。这里的 actor 权重 warm start 与 planner 复用旧候选是两件事。

## 2. 训练数据：随机 K，不输入剩余时间

复用原训练缓存中的低噪声 episodes，保持 `max_sigma=0.04` 和原有 train/val split。latent 必须来自与部署相同的冻结 encoder/projector；WM predictor 是否经历过 on-policy finetuning，不应改变这套 latent 坐标。

### 2.1 已完成的 100k：完整 16-block 锚点

以 block 边界为时间索引。第一版只从后续有完整 16 个有效动作 blocks、且相应状态帧存在的位置采当前时刻 t，避免 episode 边界改变目标 offset 分布。对每个训练样本：

```text
K ~ Uniform{1, ..., 16}             # 0.5–8 秒
当前状态 = z_t
目标状态 = z_goal = z_(t+K)         # 同一训练轨迹中真实到达的未来状态
动作标签 = [a_t, ..., a_(t+15)]     # 固定 16 blocks
```

K 只用于构造样本，不输入 actor。不能跨 episode 取 goal，也不能将其他轨迹或验证参考图像任意配给动作。训练与评估场景保持隔离；不使用评估视频、验证 rollout 或正式评估目标作训练标签。

整条 16-block chunk 共享当前 z_t 与同一个 z_goal，不在不同动作位置输入真实未来 z_(t+h)。后者在部署前不可获得。

当 K<16 时，前 K 个 blocks 通向 goal，后续 labels 是示范到达后继续发生的动作；这些仍是合法的监督。**不因为 goal 较近就把 K 之后的动作 mask 掉，也不要求第 16 步仍停在这个目标。** 原 mask 机制可以保留以兼容缓存格式，但当前采样原则下 16 个目标 blocks 都应有效。

这种采样覆盖不同远近的目标。也可理解为在轨迹内固定一个未来 goal，让当前锚点逐渐向它移动；它不表示部署时知道真实剩余时间。官方参考帧的时间戳、subgoal interval 和目标停留时间均不能直接当作实际 time-to-goal。

### 2.2 A4：覆盖不足 16 blocks 的轨迹尾部

第一版的完整窗口限制使每条轨迹最后不足约 8 秒的位置不会作为当前状态输入 actor；这些尾部动作仍可能作为更早锚点的动作标签参与训练。A4 补回这些当前状态，检查对后期操作、最后一箱及收手阶段的帮助，不预先认定它们是现有失败的原因。

对每条满足原 split 与低噪声条件的 episode，令 `s` 为起始 frame 下标，`n = min(control_steps // STRIDE, length - 1)` 为有效 pre-stop blocks 数。允许从所有 `t in {s, ..., s+n-1}` 中取锚点；在全部有效锚点之间均匀采样，包含原来的完整窗口与新补入的尾部窗口，第一版不额外过采样尾部。

```text
R = min(16, s + n - t)            # 该锚点剩余的有效监督长度，1..16
K ~ Uniform{1, ..., R}
current = z_t
goal = z_(t+K)                   # 同一 episode 中真实存在的未来状态
actions[0:R] = [a_t, ..., a_(t+R-1)]
actions[R:16] = padding           # 不作为真实动作标签
mask[i] = (i < R), i = 0..15
```

actor 仍一次输出 16 blocks，K 与 R 均不作为 actor 输入。R=16 的锚点仍使用原来的 K=1..16；尾部的 K 在剩余有效范围内均匀采样，因此 A4 总体 K 分布不再保证在 1..16 上均匀。manifest 记录这个条件分布及实际采样统计。

mask 只由有效动作长度 R 决定：当 K<R 时，K 之后仍存在的示范动作继续参与 FM 监督。禁止跨 episode 读取、将 padding 当监督、重复最后一个动作作为额外标签，或把环境 post-stop 自动 finalization 当作 teacher 执行的动作。train/val 都采用相同边界规则，各自保持原场景隔离。

## 3. Actor 输入与初始化

使用训练 split 的同一组 latent 均值 mu_z 和标准差 sigma_z：

```text
z_current = (z_t    - mu_z) / sigma_z
z_goal    = (z_goal - mu_z) / sigma_z
m         = z_goal - z_current
p         = z_current * m                 # 逐元素乘积
condition = [z_current, m, p]              # 3 × 192 = 576 维特征
```

归一化仅用于 policy 条件。WM rollout 与距离评分仍使用原来的 latent 坐标，不改变 WM 或目标距离定义。m 是目标差分，p 是手工状态–目标交互特征，不假设 latent dynamics 全局线性。

建议保留现有 observation token，使用残差目标分支：

```text
obs_token = original_obs_in(z_current) + goal_adapter([m, p])
```

goal_adapter 将 384 维输入投影到现有 Transformer hidden dim，最后一层零初始化。这样初始 actor 能保持官方 actor 的行为，再通过训练学习使用目标。保留原 action tokens、位置编码和因果 attention；所有动作位置均能读取这个目标条件。

训练原 actor backbone 与新增 adapter，WM 不加入 optimizer。目标是一个 goal-conditioned flow policy，不将 diffusion actor 替换为 Gaussian actor，也不完整复现 INTACT 的联合 representation/local/goal 训练。

新增 checkpoint 明确标记模型类型、目标特征形式与 adapter 配置，不能让旧 load_policy 静默按无目标模型加载。官方 policy.pt 保留，新模型写到独立路径。

## 4. FM loss 与训练方式

保持原 rectified-flow objective。对每个动作 block i 独立采噪声比例 s_i 与 Gaussian noise eps_i：

```text
X_i       = (1 - s_i) * A_i + s_i * eps_i
v_target  = eps_i - A_i
v_pred    = actor(condition, X, s)
loss      = 原先对有效位置聚合的 velocity MSE
```

一次 forward 同时预测所有 16 个 blocks 的 velocity，形状为 `[B, 16, 125]`；对有效动作位置计算原 FM loss。初始阶段 16 个位置均有效，A4 由 R 决定 mask。不只监督第一个 block，不增加 latent reconstruction、WM rollout、goal-distance 或 reward loss。

A4 的不同样本有效长度不同，loss 按原定义对所有有效 block 的误差求和，再除以有效 block 总数。microbatch 梯度累计也按有效 block 数归一化，使结果与同一完整 batch 的 masked FM loss 一致；validation 的动作 MSE 同样排除 padding。

输入条件、训练 goal 的构造发生变化，FM loss 的形式不变。所有 latents 可预先缓存，训练时不必运行图像 encoder 或 WM predictor。

采用官方 actor 初始化后微调，沿用原训练器的 AdamW、学习率调度与 EMA 框架。正式训练前固定并记录训练步数、有效 batch、学习率、warmup、EMA、随机种子、数据过滤和 checkpoint 选择规则；不使用 A1/A2/A3 的最终评估结果反复选择训练超参或模型。若显存不足，可使用 microbatch/梯度累计或按 batch 读取缓存，记录实际有效 batch，不因此改变 K 分布或 loss。

先验证零初始化 adapter 下与官方 actor 的 forward/采样一致性，再验证未来 goal 与 16-block 标签对齐、无跨 episode/评估数据泄漏。训练后先做少量闭环检查，FM validation loss 本身不能代替任务表现。

### 4.1 A4：原定 50k 续训预算，实际锁定累计 120k

本阶段原定预算为新增 **50,000 optimizer steps**，目标累计 **150,000**；训练使用的学习率调度以这份 50k 预算为准。实际按用户指令提前结束，并选择本阶段第 **20,000** 步的 EMA，累计 **120,000**，作为 A4 与后续 A5 的固定 actor。以下 50k/150k 数字保留为历史预算，不代表已经完成。

固定使用本批 A1/A2/A3 实际评估的 round-two final EMA 权重作为父模型，来源见 [100k 评估 manifest](../../outputs/planning/v2/round2-final-20261104/run.json)：

```text
parent checkpoint (cluster):
  /mnt/projects/ll/tianzizh/cis6280-v2-eval-r2-20261104/checkpoints/goal-actor-r2-final.pt
SHA256:
  d039fbba25eabb9b976fff945afd6b59031cbbf11df5f916a5671dd774e234bf
parent cumulative optimizer steps: 100000
new optimizer steps:                50000
final cumulative optimizer steps:  150000
```

这里的 100k/50k 都指 optimizer steps，不是遍历全部数据的 epochs。完整加载父模型 backbone 与已学到的 goal adapter，不重新零初始化 adapter；沿用父模型的架构、latent/action normalizers、数据 split 与 `max_sigma=0.04`，只改变第 2.2 节的训练样本覆盖。

新建一个 50k 阶段，沿用 round-two 的优化配置：AdamW，lr=1e-4，betas=(0.9, 0.95)，weight_decay=0.01（沿用原参数分组），warmup=1000，随后在本阶段 50k 内 cosine decay；有效 batch=256、microbatch=64，gradient clip=1.0，EMA=0.999，seed=0。训练 actor backbone 与 goal adapter，encoder/projector、WM action encoder 和 predictor 保持冻结。

warm start 加载父 EMA actor 权重；新建 optimizer 与 LR schedule，EMA 初值也来自这份父 actor。按新阶段 seed 初始化数据与 flow RNG，明确记录初始化方式。该阶段中断后的恢复使用其自己的完整 training state，恢复 optimizer、scheduler、EMA 与 RNG，不重复执行权重 warm start。

原定选择新阶段第 50000 步的 final EMA（累计 150k）；本次实际使用用户指定的第 20000 步 EMA（累计 120k），选择原因记录为 `user_selected_intermediate_ema`。不依据 A4/A5 任务结果另挑 actor checkpoint。新建 checkpoint、日志与评估目录，父 100k checkpoint 和已有 A1/A2/A3 结果保持独立。

本次训练运行：Job `1502354`，共享 `batch` 队列，节点 `dj-a40-0.grasp.maas`，单张 NVIDIA A10 24GB，bfloat16；liu 队列预计等待更久，因此使用共享 GPU。训练工作区为 `/mnt/projects/ll/tianzizh/cis6280-v2-a4-20261104`，日志和 checkpoint 使用 run 名 `goal-actor-v2-a4-tail-seed0-20261104`。[W&B 训练记录](https://forge.coreweave.com/wandb/tianzizh-university-of-pennsylvania/6280_actor/runs/v2actor-20261104-a4-tail-s0)，[本地启动配置](../../outputs/training/v2/goal-actor-v2-a4-tail-seed0-20261104/launch.json)。A4-16/A4-32 的正式评估仍需按第 7.2 节匹配已有 A2 的 L40、dtype、workers 和视频设置。

### 4.2 A5：冻结 A4 actor 与图像表示，微调当前策略的 WM

目标是检验：在同一份 A4 actor 生成的候选上，适配当前状态/动作分布的 WM 是否能改善候选排序与真实任务表现。A5 只改变 WM 的动力学预测部分，A5-16/A5-32 使用同一份微调后的 WM，分别与 A4-16/A4-32 比较。

**固定模型与更新范围。** actor 使用 A4 实际锁定的权重，来源见 [A4 测评准备记录](../../outputs/planning/v2/a4-120k-20261104/preparation.json)：

```text
actor checkpoint (cluster):
  /mnt/projects/ll/tianzizh/cis6280-v2-a4-20261104/checkpoints/goal-actor-v2-a4-tail-seed0-20261104/step-020000.pt
actor cumulative optimizer steps: 120000
WM initialization: A4 评估实际使用的同一份 world-model checkpoint
```

采集前记录并锁定 actor 与原 WM 的 SHA256。WM 从 A4 所用权重继续微调，不退回 Stage 2 的 pretrain.pt；官方 checkpoint 保留，微调结果另存。

| 模块 | 采集期间 | WM 微调期间 | A5 评估期间 |
| --- | --- | --- | --- |
| A4 actor backbone 与 goal adapter | 冻结 | 冻结 | 冻结 |
| 图像 encoder 与 projector | 冻结 | 冻结 | 冻结 |
| WM action_encoder / predictor / pred_proj 的可训练层 | 原 WM，冻结 | 更新 | 新 WM，冻结 |
| 其他模块与 normalizers | 保留原值 | 保留原值 | 保留原值 |

图像 encoder/projector 始终使用 eval mode，参数和包括 BatchNorm running statistics 在内的 buffers 都不改变；沿用 Stage 4 的做法，pred_proj 中的 BatchNorm 保持 eval mode。训练前后核对 actor、encoder/projector 的参数与 buffers 身份，保证原 actor 条件、已有 latent cache、goal latents 与阈值仍处于同一坐标系。

**采集当前策略的真实分支结果。** 在官方 train split 内，预先按 scene 划分互斥的 WM-train 与 WM-calibration 清单；calibration 用于选择 WM checkpoint。两份清单都不包含本轮 20 个开发评估 scenes 或预留正式验证 scenes，同一 scene 的所有 seeds 与分支留在同一集合。保持 actor 原有训练数据 split 不变，A5 另记自己的采集清单。

goal actor 在训练场景也必须接收相应的参考目标。为这些采集 scenes 用 scripted reference 生成同样每 80 control steps（8 秒）的参考图片，使用冻结 encoder 编码；可以复用 [goals.py](../../assignment/training/planning/goals.py) 的 train-split 生成能力。这些训练用参考序列单独保存，正式评估仍使用原官方 goals 文件。

主轨迹由固定 A4 actor、原 WM 和原 A2 评分驱动；A4-16 与 A4-32 两种行为配置各占一半采集预算，覆盖两组会访问的状态。每轮只执行首 block，真实观测驱动目标切换。全程采样锚点，保留正常操作与抓空、掉箱等物理失败，不只采成功 episode 或早期状态。

在抽中的 block 边界，保存完整 simulator 状态和控制器相关状态，记录真实 WM context 及对应的已执行动作；以当时的 active goal 为条件，独立采样候选。每个候选都从同一 snapshot 开始，连续执行完整 16-block plan，每个 block 后渲染并用冻结 encoder 得到真实 latent。分支内不重新规划，也不以新 goal 重新生成动作。执行完恢复 snapshot，再继续原主轨迹；分支采样使用独立 RNG，不改变主轨迹的 policy RNG 或历史。

第一版采集目标如下，具体 scenes、seeds、锚点抽样率和实际数量在正式运行前写入 manifest：

| 数据用途 | 锚点目标数 | 每锚点候选数 | 每候选动作长度 |
| --- | ---: | ---: | ---: |
| WM-train | 5000 | 16 | 16 blocks |
| WM-calibration | 500 | 32 | 16 blocks |

首批以具有完整 16-block 合法执行预算的锚点/分支为准。物理任务失败仍是有效动力学标签；数值异常、时限截断或环境自动 finalization 导致的不完整分支单独记录并排除，不用重复帧或 WM 预测补足真实标签。若完整锚点不足，按预先固定的训练场景顺序补采，并记录实际预算。

每个锚点至少保存 scene/seed、control step、active goal、context latents、历史动作、候选动作和各 block 后的真实 latents。候选 raw actions 先按部署方式裁剪到 [-1,1]，再用原 action mean/std 重新归一化；WM 训练、历史及评估使用这一份实际执行动作，不能用裁剪前的动作配裁剪后的结果。

**训练目标与预算。** 固定 actor 与图像表示，仅训练 action_encoder、predictor 和 pred_proj 的可训练部分。输入真实 context 和候选动作，autoregressive open-loop 预测 16 个未来 latents，后续预测使用前面的 imagined latents；向全部 16 步反向传播：

```text
z_true[n,h] = frozen_encoder(real_frame_after_candidate_n_block_h)
z_hat[n,1:16] = WM_theta(real_context, executed_normalized_actions[n,1:16])
L_WM = mean_{n,h,latent_dim} (z_hat[n,h] - stopgrad(z_true[n,h]))^2
```

goal 不作为 WM 新输入；训练监督为真实未来状态，不增加将预测直接拉向 subgoal 的 loss，不使用 imagined latents 充当真实标签，也不更新 actor 的 FM loss。

第一版沿用现有 Stage 4 recipe：60000 optimizer steps，batch=256 个候选轨迹，AdamW lr=1e-4、betas=(0.9,0.999)、weight_decay=1e-3，warmup=1000 后 cosine decay，gradient clip=1.0，seed=0；从原 WM 权重初始化新 optimizer/scheduler。若使用 microbatch，保持有效 batch 与完整 16-step loss 相同。固定一批采集数据完成一次 finetuning，不在本轮混入重新采集、再训练的多轮迭代。

**WM checkpoint 的选择必须匹配 A2/A4。** 现有 Stage 4 验证按终点距离排序，需要改为第 6.1 节的全轨迹最小距离。对同一 calibration 锚点的 32 条候选，分别在前 16 条和全部 32 条上计算：

```text
C_pred[n] = min_{h=1,...,16} ||z_hat[n,h] - active_goal||^2
C_true[n] = min_{h=1,...,16} ||z_true[n,h] - active_goal||^2
pick = argmin_n C_pred[n]
regret = (C_true[pick] - min_n C_true[n]) / (std_n(C_true[n]) + 1e-8)
```

记录 h=1/4/8/16 的 latent MSE、top-1 agreement 与 ranking regret。只在完整有效的同一候选集合上比较原 WM 和新 WM；真实 min-distance 最优候选用于校准排序，不等同于真实任务最优动作。

在 step 0 和每 2000 optimizer steps 评估固定 calibration 集，以两个候选规模的 mean regret 平均值 `(regret_16 + regret_32) / 2` 选择一份共用 WM；相同时保留较早 checkpoint。step 0 的原 WM 也参与选择，如无改善则保留它并如实报告。选择过程不查看 A5 开发/正式任务回报，不为 A5-16 与 A5-32 各自选择不同模型。

已有 [collect_onpolicy.py](../../assignment/training/data/collect_onpolicy.py) 和 [finetune_onpolicy.py](../../assignment/training/training/finetune_onpolicy.py) 可复用框架，但采集器需要接入 goal actor、训练场景参考目标与 A4 主轨迹；训练器需要适配上述排序验证、数据清单和模型冻结核对。当前官方 on-policy 数据来自旧 actor，不能直接作为本轮新采集数据的替代。

## 5. 部署统一接口

A1/A2/A3/A4-16/A4-32/A5-16/A5-32 的 actor 均接收当前真实观测的 latent 和 **当前 active goal** 的 latent，以及由两者构造的 m/p；不输入 K、R、剩余时间或 next goal。A3 的 next goal 只用于 planner 评分。

所有组先根据新真实观测运行官方目标切换：

```text
T_m = 0.25 * ||g_m - g_(m-1)||²
若 ||z_t - g_m||² < T_m，则按官方规则推进 active index
```

首个目标的前一状态为该 episode 官方 init frame 的编码。T_m 使用参考序列相邻状态定义，不随本轮候选变化。最后一个目标没有后继，不继续越界。

实际 active index 仅根据真实观测更新；imagined rollout 达标不切换真实目标。没有 timeout 强制前移、目标回退或补采循环。

## 6. 对照矩阵

| 组别 | Actor | 候选数 | WM predictor | 动作选择 |
| --- | --- | ---: | --- | --- |
| A0：公平的旧 actor 对照 | 官方无 goal actor | 16 | 使用 | 全 16 步到 active goal 的最小距离 |
| A1：Direct | 累计 100k goal-conditioned actor | 1 | 不使用 | 直接执行该候选首 block |
| A2：Active-goal planning | 与 A1 相同 100k checkpoint | 16 | 使用 | 全 16 步到 active goal 的最小距离 |
| A3：Active-first lookahead | 与 A1 相同 100k checkpoint | 16 | 使用 | 当前目标优先，再看终点到 next goal 的进展 |
| A4-16：尾部覆盖续训 | 当前锁定的累计 120k EMA | 16 | 原 WM | 沿用 A2，全 16 步到 active goal 的最小距离 |
| A4-32：尾部覆盖续训 | 与 A4-16 相同 120k checkpoint | 32 | 原 WM | 沿用 A2，全 16 步到 active goal 的最小距离 |
| A5-16：On-policy WM | 与 A4 完全相同且冻结的 120k actor | 16 | A5 微调并锁定的 WM | 与 A4-16 相同评分 |
| A5-32：On-policy WM | 与 A5-16 相同 actor | 32 | 与 A5-16 相同 WM | 与 A4-32 相同评分 |

A0 仍用 encoder 与 planner 的 active goal，但 actor 自身不看 goal。它与 A2 只有 actor 不同，避免将旧 G1 的尾窗评分和新 actor 同时改变后，无法解释收益。A2−A0 衡量新 actor 在相同 planner 下的整体收益，不单独证明乘积项的因果贡献。

A1 的一个 candidate 是一次完整 flow 采样，不是多条候选的均值。A1 仍使用冻结 encoder 编码当前图像和目标，只是不调用 WM predictor。各组每轮均只执行选中计划的第一个 block，之后重新观测、生成和选择。

### 6.1 A0 / A2 / A4 / A5：全轨迹最小距离

记候选 n 在第 h 个 block 后的 imagined latent 为 z_hat[n,h]：

```text
D_m[n,h] = ||z_hat[n,h] - g_m||²
C_active[n] = min_{h=1,...,16} D_m[n,h]
best = argmin_n C_active[n]
```

只包括未来 16 个预测 latents，不把当前真实 z_t 加入 min；选择计划后仍从它的第一个 block 开始执行，不能直接跳到最接近目标的动作位置。cost 完全相同时保留较早采样的候选。

该评分允许目标在 chunk 中途到达，与随机 K 的训练语义一致。它不保证预测到达后持续保持目标，需结合真实任务结果评价。

### 6.2 A3：当前目标优先的两目标评分

A3 使用同一轮的 16 条候选和一次 16-step WM rollout，同时计算 active/next 两种距离，不额外调用 predictor。actor 仍只以 active goal 为条件。

先寻找预测在第 1–15 步内到达当前目标的候选，留至少一步表示后续进展：

```text
eps = 1e-8
q[n] = min_{h=1,...,15} D_m[n,h] / max(T_m, eps)
r[n] = ||z_hat[n,16] - g_(m+1)||² / max(T_(m+1), eps)
eligible = {n : q[n] < 1}
```

选择规则：

1. 已是最后一个 subgoal 时，使用 A2。
2. eligible 为空时，使用 A2，包括完整第 1–16 步的 min。
3. 否则，只在 eligible 中最小化 `J[n] = q[n] + 0.25*r[n]`，平分时保留较早候选。

0.25 是本轮固定的 next-goal 权重，与官方到达阈值中的 tau 是两个独立参数，分别命名并记录。距离按各目标的官方阈值归一化，减轻两个目标原始距离量级不同的问题。

不要求 next goal 也在 8 秒内完成；r 衡量向后继目标的进展。资格筛选防止仅因终点接近 next goal 就跳过 active goal。当前目标只在第 16 步达标、此前未达标时，不强行做 lookahead，按 A2 处理。

新 actor 的训练 chunk 中可以包含到达 active goal 后的继续行为，A3 从这些延续候选中选择；但 next goal 未输入 actor，不能保证候选充分覆盖它。记录 eligible 非空、实际启用 lookahead 的比例，用于区分方法未触发与触发后无收益。

### 6.3 A4-16 / A4-32：相同新 actor，不同候选数

两组使用第 4.1 节实际锁定的累计 120k `step-020000.pt` actor EMA。每次分别一次性采样 16 或 32 条完整 16-block plans，用原 WM 预测，并严格采用第 6.1 节的 A2 评分；没有 next-goal lookahead、短视野评分、连续多帧评分、补采样或跨轮候选复用。

沿用官方每 8 秒的 subgoal 图片、真实观测目标切换 tau=0.25、generate=16、execute=1、n_steps=10、lag=0，以及当前 A2 的动作反归一化、裁剪、供 WM 使用的重新归一化与历史记录方式。A4-16 和 A4-32 之间只有候选数不同。

### 6.4 A5-16 / A5-32：固定 A4 actor，更换 WM

只将 A4 的原 WM 替换为第 4.2 节锁定的一份 on-policy finetuned WM。两组在 actor、图像 encoder/projector、normalizers、目标序列、阈值、评分、采样与执行方式上均与同规模 A4 相同。推理期间所有模块冻结，simulator 分支执行仅用于训练数据采集，不进入正式 planner 决策。

主要对照为 A5-16−A4-16 和 A5-32−A4-32，分别衡量同一 actor、相同候选预算下更换 WM 的闭环收益；A5-32−A5-16 作为新 WM 下增加候选数的辅助比较。不同 WM 会改变闭环选择及后续访问状态，不要求两组真实轨迹一直相同。

## 7. 实验顺序与比较

### 7.1 已完成的 100k 开发评估

本批 A1/A2/A3 各 60 条，共 180 条，使用固定 20 scenes × sampler seeds 0/2/3，结果见 [summary.json](../../outputs/planning/v2/round2-final-20261104/summary.json)。这批没有运行 A0，A0 仍保留为原设计中的公平旧 actor 对照，不能将已有跨批次结果当作这批 A0。

| 组别 | mean peak | mean final | Success |
| --- | ---: | ---: | ---: |
| A1 | 2.48 | 1.98 | 20.0% |
| A2 | 2.72 | 1.98 | 21.7% |
| A3 | 2.52 | 1.77 | 15.0% |

A2 达到至少 3 箱为 39/60，达到至少 4 箱为 16/60；A1 分别为 25/60 与 16/60。A3 未优于 A2，scene 配对 bootstrap 差异区间仍包含零。本轮直接进入尾部覆盖续训，不将新的 planner 搜索改动混入 A4。

### 7.2 A4 的执行记录与对照

1. 实现第 2.2 节的尾部锚点、合法 K 和动作 mask；检查 R=1、R=16 与 episode 边界，并验证梯度累计的有效 block 归一化、validation padding 排除及数据隔离。保留 100k 阶段配置与结果，明确记录新的采样版本。
2. 父模型为累计 100k actor，按第 4.1 节的原定 50000-step 调度启动；实际在用户停止后固定第 20000 步 EMA，累计 120k，保留独立训练证据及恢复状态。
3. 锁定累计 120k 的 `step-020000.pt`，检查 A4-16/A4-32 的候选维度、评分和首 block 执行；实现检查不用于挑选权重或训练 checkpoint。
4. 沿用本批 [20-scene 清单](../../outputs/planning/v2/round2-final-20261104/scenes-20.json) 和 sampler seeds 0/2/3，运行 A4-16、A4-32 各 60 条，共新增 120 条。固定同一 WM、goals、环境、GPU/dtype、workers 与视频设置。满足相同运行条件时复用已完成的 A2 作为配对参考；条件变化则补跑同条件 A2。
5. 比较 A4-16−A2（尾部覆盖及实际额外 20k 训练的整体收益）、A4-32−A4-16（同一新 actor 下增加候选数的收益）。报告任务表现及决策成本，再锁定用于正式验证的组别。

本轮两组不能独立证明尾部覆盖的因果贡献：A4-16 相对 A2 同时增加了实际 20k 训练。若后续需要隔离两者，再加入原完整窗口采样续训相同步数的对照；它不属于此次 A4-16/A4-32 的必跑矩阵。

原设计中的 A2−A0（同 planner 下的新旧 actor）、A2−A1（新增候选与 WM 筛选的整体收益）、A3−A2（next-goal lookahead）仍按各自可用证据报告，不与 A4 的新训练配置混算。

### 7.3 A5 的执行顺序

1. 固定与 A4 相同的 120k actor 和原 WM，记录哈希；预先固定 WM-train/WM-calibration 场景、训练用 8 秒参考目标、采集预算与 RNG，确保评估场景不进入数据采集或 checkpoint 选择。
2. 完成 goal actor 采集器适配，检查 snapshot 恢复、实际执行动作与监督 latent 对齐、冻结图像表示及 A4-16/32 两种行为覆盖；再完成第 4.2 节的整批采集。
3. 从同一原 WM 初始化，按固定 recipe 训练一次，使用独立 calibration 的全轨迹 ranking regret 选择同一份 WM。保存训练状态、原/新 WM 的预测及排序指标、冻结模块核对结果，随后锁定 checkpoint。
4. 固定相同 actor、新 WM 和 planner 配置，运行 A5-16、A5-32 各 20 scenes × seeds 0/2/3，各 60 条，共新增 120 条。与 A4 使用相同环境、GPU/dtype、workers 和视频设置；满足条件时复用 A4 的两组结果，条件变化则补跑同条件 A4。
5. 分别报告 A5-16−A4-16、A5-32−A4-32 的任务表现、scene 配对区间与决策成本；预测 MSE 或 calibration 排序改善只作为机制证据，不单独宣布闭环任务提升。锁定方法后再做预留正式验证，不根据开发/正式任务结果重选 WM。

v1.1 的第二批 20 个固定验证场景也已经看过结果，不能在 v2 再称为全新测试集。正式 v2 验证需先从剩余官方场景固定一份未用于本轮调参的清单，锁定 actor/checkpoint、权重和部署配置后再评估；不根据验证结果重新训练或选择参数。

A2 不一定优于 A1。新 actor 改变了候选分布，冻结 WM 对这些动作的预测/排序可能失准；A2 退步时应分别检查 actor 与 selector，而不是直接判定 goal conditioning 无效。

### 7.4 A5 实际结果：本轮未取得收益

5000/500 个锚点的采集与 60k-step WM 微调已完成。预设的 `(regret_16 + regret_32)/2` 在原 WM 上为 **0.50420**，最佳微调 checkpoint（56k）为 **0.50724**，因此仍保留 step 0。用户随后额外指定 56k WM 跑 A5-32，各 60 条配对开发测评中，mean peak/final boxes 从 A4-32 的 **2.87/2.13** 降至 **2.58/1.83**，success 从 **14/60** 降至 **11/60**，平均决策耗时均约 **217 ms**。

记录结论为本轮 on-policy WM finetuning 效果不好、没有观察到闭环收益；scene 配对区间包含或触及 0，不能将平均下降表述为已证明稳定退步。原选模仍为 step 0，56k 属于独立补充实验；未运行微调 WM 的 A5-16 或预留正式验证集。数据、配置、区间、原始指标和集群证据路径见 [A5 RESULTS.md](a5/RESULTS.md)。

## 8. 主要指标和必要记录

以真实任务结果为主：mean peak/final boxes、达到至少 1/2/3/4 箱的无条件比例、官方 success、第一箱到第二箱的转化及时间、mean/P95 决策耗时。A4/A5 额外汇总第三箱到第四箱的转化及时间，关注改进是否延伸到后期；未到下一阶段者保留截尾观察，不仅报告成功者耗时。不同 seed 的同一 scene 属于一组，配对 bootstrap 按 scene 重采样。

沿用官方 stack 计数语义；手接触等会影响有效箱数，final 低于 peak 不直接代表倒塌。A3 目标进展或更低的 latent cost 仅用于解释机制，不能代替任务收益。

轻量记录足以支持本轮问题：

- 每轮 active index、真实目标距离/阈值和实际切换。
- A0/A2/A4-16/A4-32/A5-16/A5-32 的候选全轨迹最小距离、选中 ID 与最近 block 时刻，以及实际候选数。
- A3 的 q/r、eligible mask、实际 selection cost、是否回退及选中 ID。不能用“选中候选必须具有最小 current distance”的旧断言检查 A3。
- 各组的总决策耗时，encoder、policy 与 predictor 调用次数；A1 的 predictor 次数应为零。
- 代表性 episode 的视频，至少展示真实画面、active goal、任务进度；A3 可以显示 next goal。
- A5 独立 calibration 上原/新 WM 的 h=1/4/8/16 MSE、16/32 候选的 agreement/regret 与实际选择的 WM step；采集和训练成本单独报告，不混入在线决策耗时。

不常规保存所有候选动作或完整 latent 张量；必要时只为少量诊断案例保存。没有动作裁剪等细节的单独 ablation，也不要求大规模逐帧人工标注。

## 9. 输出与复现

```text
checkpoints/<goal-actor-run>/
  policy.pt
  run.json
outputs/training/v2/<train-run>/...
outputs/planning/v2/<eval-run>/
  run.json
  summary.json
  A0/episodes/scene-<scene>.sample-<seed>/...
  A1/episodes/...
  A2/episodes/...
  A3/episodes/...
  A4-16/episodes/...
  A4-32/episodes/...
  A5-16/episodes/...
  A5-32/episodes/...
  analysis/
dataset/onpolicy-v2/<collect-run>/
  manifest.json
  wm-train-scenes.json
  wm-calibration-scenes.json
  train-*.npz
  calibration-*.npz
checkpoints/<a5-wm-run>/
  model.pt
  train-state.pt
outputs/training/v2/<a5-wm-run>/...
```

训练 manifest 保存代码版本与工作区状态、数据 split/过滤、latent cache 及 encoder 标识、mu/std、K 分布、有效锚点数、actor 初始化哈希、优化配置和 checkpoint 选择规则。A4 另记完整/尾部锚点数量、R 与 K 的实际采样分布、mask/归一化语义、父 checkpoint 哈希与累计 100k、原预算本阶段 50k/累计 150k、实际所选本阶段 20k/累计 120k，以及 optimizer/scheduler/EMA/RNG 的初始化及恢复方式。新 checkpoint 保存全部目标接口配置，确保评估按正确架构加载。

A5 的采集 manifest 保存固定 actor/原 WM/encoder 身份、训练参考 goals 哈希、场景集合、两种行为比例、采集与候选 RNG、完整锚点/分支数量、异常排除统计及实际动作归一化规则。WM 训练 manifest 另记 shard 哈希、训练预算、更新模块、冻结参数/buffers 核对、step-0 与各 checkpoint 的 calibration 指标及统一选模规则；新 WM 必须可以与原 A4 actor、normalizers 和目标编码直接配合加载。训练数据、原 WM 与新 WM 分别保存，不覆盖 A4 的模型或证据。

评估 manifest 保存官方 WM、原 actor、新 actor 和 goals 的 SHA256，完整场景/seed、A3 权重/阈值定义、环境版本、GPU/dtype、采样设置及训练 run/checkpoint 来源。不修改官方 checkpoint，不覆盖 v1.1 的运行目录或证据。若评估时推理环境不同，补齐同环境对照后再比较。

已有 v2 实现入口：[goal_policy.py](../../assignment/training/training/goal_policy.py)、[train_goal_policy.py](../../assignment/training/training/train_goal_policy.py)、[goal_actor.py](../../assignment/training/planning/goal_actor.py) 和 [evaluate_goal_actor.py](../../assignment/training/planning/evaluate_goal_actor.py)。A5 新入口为 [collect_goal_onpolicy.py](../../assignment/training/data/collect_goal_onpolicy.py)、[finetune_goal_onpolicy.py](../../assignment/training/training/finetune_goal_onpolicy.py) 和 [onpolicy_a5.py](../../assignment/training/training/onpolicy_a5.py)，复现及恢复说明见 [A5.md](../../assignment/training/training/A5.md)。WM 训练注册到 W&B project `6280_wm`，固定 run ID `a5-wm-onpolicy-120k-s0`。

首批采集配置已锁定为 128 个 WM-train scenes 与 32 个 WM-calibration scenes（均属官方 train split），seeds 0/2/3，锚点概率 0.25；两种行为各占 5000/500 锚点预算的一半。[固定采集计划](a5/collection-plan.json) 记录实际模型 SHA、场景顺序与 RNG，[正式保留场景](a5/reserved-scenes.json) 从官方 goals 中排除本轮 20 个开发 scenes 和 v1.1 已看过的第二批 20 scenes 后固定，详见 [选择证据](a5/reserved-scenes-provenance.json)。该保留集尚未运行。

## 10. 完成标准

保留已完成的 100k A1/A2/A3 结果、A0 对照设计和 A4 尾部训练记录；A4 两组使用当前固定的累计 120k actor。A5 完整冻结这份 actor 与 encoder/projector，独立采集当前策略的数据并微调 WM，锁定同一份新 WM 完成 A5-16/A5-32 各 60 条开发评估，分别与同候选数的 A4 比较真实任务、成本和视频结果。通过固定验证集检查开发结果的可复现性，不以更低的 FM/预测 loss、calibration cost 或单条成功视频宣布方法有效。
