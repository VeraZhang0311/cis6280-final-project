# A4-32 HOLD / LOCAL-HOLD Results

Updated November 10, 2026 (America/New_York). Only the Colab screen for scene
`100038`, sample seed `0`, is complete. The matched L40/BF16 evaluation over 20
scenes and sample seeds `0/2/3` has not been run. This document therefore records
mechanism evidence and the next evaluation decision; it does not claim a
generalizable improvement.

## Changes

All three groups use the same v2 120k EMA goal actor, original frozen world
model, 32 candidates, 16-block rollout, ten flow-velocity evaluations,
real-observation-only goal switching, and execution of only the first 0.5-second
block before replanning.

| Method | Before latch | After latch | Recovery rule |
| --- | --- | --- | --- |
| `A4-32` | Minimum active-goal distance over all 16 blocks | Unchanged | Not applicable |
| `A4-32-HOLD` | Exactly A4-32 | Minimize mean goal distance over blocks 13--16 | Permanently latched |
| `A4-32-LOCAL-HOLD` | Exactly A4-32 | If all first four blocks are within threshold, choose lexicographically by first-block wrist XYZ RMS, full-action RMS, and candidate index; otherwise minimize mean distance over the first four blocks | Return to A4-32 after two consecutive real observations outside the threshold |

Both HOLD variants enter only at the final official goal, after two consecutive
real observations satisfy the strict condition
`distance < 0.25 * reference_gap`. The planner does not read simulator box
counts, rewards, contacts, or success.

## Single-scene screen

Evidence directory:
`/content/drive/MyDrive/cis6280-final-project-v2/outputs/a4-32-local-hold/20261111-015436`.

| Method | Peak | Final | Official success | Four-box retention | Mean decision ms | Decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A4-32 | 4 | 1 | 0 | 0/1 | 188.2 | 359 |
| A4-32-HOLD | 4 | 1 | 0 | 0/1 | 184.7 | 359 |
| **A4-32-LOCAL-HOLD** | **4** | **4** | **1** | **1/1** | 185.2 | 275 |

LOCAL-HOLD reached official success and terminated early, using 84 fewer
planning decisions than either failed group, equivalent to approximately 42
seconds of simulator time. Mean decision latency was similar across the three
groups, so this screen showed no inference-time penalty for LOCAL-HOLD.

## HOLD mechanism audit

| Method | Latched decisions | Entries | Exits | Decisions with a safe candidate | Selected-candidate wrist RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| A4-32 | 0 | 0 | 0 | Not applicable | Not applicable |
| A4-32-HOLD | 87 | 1 | 0 | Not applicable | Not applicable |
| A4-32-LOCAL-HOLD | 3 | 1 | 0 | **0/3 (0%)** | 0.0078 |

None of the three latched LOCAL-HOLD decisions had a candidate whose first four
imagined blocks all remained within the threshold. All three decisions therefore
used the `first4_mean` fallback. The low selected-candidate wrist RMS is an
observed property of the fallback choices, not an effect of the wrist-ranking
branch.

## Conclusions supported by current evidence

- The original late-four HOLD did not help in this scene: like the baseline, it
  reached four boxes, finished with one, and did not succeed. Earlier step-level
  auditing also showed that it lost its first four-box state sooner.
- LOCAL-HOLD improved the final count from one to four and converted this screen
  case to official success.
- The narrow supported conclusion is that **the first-four-block mean-distance
  fallback was better than the blocks-13--16 retention score in this scene.**
- This result does not validate strict safe-candidate filtering or wrist-RMS
  ranking, because that branch never executed.
- This is one scene/seed already used during method development. It provides no
  variance estimate or confidence interval and cannot establish that LOCAL-HOLD
  is consistently better than A4-32.

## Frozen follow-up evaluation

The next evaluation keeps implementation snapshot
`257231e8fb38df236ee63216ac18c8971fa9d77e` unchanged and reruns all three groups
under conditions matched to the v2 integration: the same 20 development scenes
and sample seeds `0/2/3`, one L40, bfloat16, eight workers, and a 1795-step
horizon. Each group has 60 episodes, for 180 episodes total. A fresh A4-32
control is included. Paired differences are aggregated across the three seeds
within each scene, then bootstrapped by scene.

Primary metrics are official success, mean final boxes, and
`P(final=4 | peak=4)`. The report also includes peak boxes, third-to-fourth
conversion, HOLD entry and exit counts, safe-candidate rate, fallback rate,
decision latency, and decisions saved by early termination. A4-32 remains the
recommended configuration until this evaluation is complete.
