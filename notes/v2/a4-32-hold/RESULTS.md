# A4-32 HOLD / LOCAL-HOLD 实验结果

更新日期：2026-11-10（America/New_York）。当前只有 scene `100038`、sample
seed `0` 的 Colab screen 完成；20 scenes x seeds `0/2/3` 的等效 L40/BF16
配对评测尚未完成。因此这里记录机制证据和下一阶段决定，不宣称泛化提升。

## 修改内容

三组固定使用同一个 v2 120k EMA goal actor、原始冻结 WM、32 candidates、
16-block rollout、10 次 flow velocity evaluation、真实观测 goal switching，
并且每轮只执行第一个 0.5 秒 block。

| 方法 | latch 前 | latch 后 | 恢复规则 |
| --- | --- | --- | --- |
| `A4-32` | 16 blocks 内最小 active-goal distance | 不变 | 不适用 |
| `A4-32-HOLD` | 与 A4-32 完全相同 | 最小化 blocks 13--16 的平均 goal distance | 永久 latch |
| `A4-32-LOCAL-HOLD` | 与 A4-32 完全相同 | 若前四 blocks 全在 threshold 内，按首 block wrist XYZ RMS、全动作 RMS、candidate index 字典序选择；否则最小化前四 blocks 平均 distance | 连续两次真实观测离开 threshold 后返回 A4-32 |

两个 HOLD 都只在最终 official goal 上、连续两次真实观测满足严格
`distance < 0.25 * reference_gap` 后进入。planner 不读取模拟器 box count、
reward、contact 或 success。

## 单场景 screen

证据目录：
`/content/drive/MyDrive/cis6280-final-project-v2/outputs/a4-32-local-hold/20261111-015436`。

| 方法 | Peak | Final | Official success | 四箱保留 | 平均决策 ms | 决策数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A4-32 | 4 | 1 | 0 | 0/1 | 188.2 | 359 |
| A4-32-HOLD | 4 | 1 | 0 | 0/1 | 184.7 | 359 |
| **A4-32-LOCAL-HOLD** | **4** | **4** | **1** | **1/1** | 185.2 | 275 |

LOCAL-HOLD 达成 official success 并提前结束，比两个失败组少 84 次规划
决策（约 42 秒仿真时间）。三组平均决策耗时接近，本 screen 没有看到
LOCAL-HOLD 的推理开销惩罚。

## HOLD 机制审计

| 方法 | latched decisions | entries | exits | 存在安全候选的比例 | 选中候选 wrist RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| A4-32 | 0 | 0 | 0 | 不适用 | 不适用 |
| A4-32-HOLD | 87 | 1 | 0 | 不适用 | 不适用 |
| A4-32-LOCAL-HOLD | 3 | 1 | 0 | **0/3 (0%)** | 0.0078 |

LOCAL-HOLD 的三次 latch 决策均没有候选满足“前四个 imagined blocks 全部
在 threshold 内”，因此三次全部走 `first4_mean` fallback。低 wrist RMS
是 fallback 选中候选的观测属性，不是 wrist 排序分支造成的结果。

## 当前可支持的结论

- 原 late-four HOLD 在该场景没有收益：与 baseline 同为 peak 4、final 1、
  success 0；此前逐步审计还显示它更早丢失第一次四箱状态。
- LOCAL-HOLD 在该场景将 final 从 1 提升到 4，并转换为 official success。
- 该结果支持的窄结论是：**完成阶段的前四-block平均距离 fallback 在这个
  场景优于 blocks 13--16 的长期 retention score。**
- 该结果不支持“严格安全候选筛选有效”或“最小 wrist RMS 导致成功”，因为
  对应分支从未执行。
- 只有一个已经用于方法开发的 scene/seed，没有方差和置信区间；不能据此
  宣称 LOCAL-HOLD 稳定优于 A4-32。

## 冻结的后续评测

下一步保持 snapshot `257231e8fb38df236ee63216ac18c8971fa9d77e` 的方法
不变，按照 v2 integration 的等效条件重跑三组：20 个相同 development
scenes x seeds `0/2/3`，L40、bfloat16、8 workers、1795 steps，每组 60
episodes，共 180 episodes。使用 fresh A4-32 control，按 scene 聚合三个
seed 后 bootstrap 配对差值。

主指标为 official success、mean final boxes 和
`P(final=4 | peak=4)`；同时报告 peak、third-to-fourth conversion、HOLD
触发/退出次数、safe eligible rate、fallback rate、决策时间和提前终止节省。
在这批结果完成前，A4-32 仍是推荐配置。
