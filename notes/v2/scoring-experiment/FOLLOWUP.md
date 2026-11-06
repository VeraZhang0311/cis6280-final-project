# T1 retention follow-up experiment

Date: 2026-11-06. Status: implementation and deterministic tests complete;
closed-loop GPU results pending.

## Motivation

The locked single-scene screen showed the same four-box peak for v2 A4-32 and
T1, but final counts of one and three respectively. T1 therefore improved
retention without producing official success. T2 showed that simply doubling
the block-16 coefficient to 0.50 was harmful. This follow-up tests whether the
remaining failure is caused by a noisy single long-horizon endpoint and whether
an explicitly bounded final-goal hold state prevents post-completion damage.

## Locked controls

Every group uses the released 120k A4 goal actor, original frozen WM, official
dense goals, 32 independently sampled candidates, 16 generated and imagined
blocks, ten flow-velocity evaluations, real-observation-only goal switching,
one executed block, and the same policy seed. No model parameter is updated.

| Group | Retention term or state change |
| --- | --- |
| `A4-32` | v2 minimum distance over blocks 1–16. |
| `A4-32-T1` | Retained T1: arrival/time score plus `0.25 * d[16]`. |
| `A4-32-T1-L4` | Replace `d[16]` with `mean(d[13:16])`. |
| `A4-32-T1-H8` | Replace `d[16]` with `d[8]`. |
| `A4-32-T1-H12` | Replace `d[16]` with `d[12]`. |
| `A4-32-T1-L4-HOLD` | Use T1-L4 normally; after hold confirmation, minimize only `mean(d[13:16])`. |

All distances are normalized by the active reference gap. L4, H8, and H12 keep
the T1 arrival-time weight `0.05` and retention weight `0.25`. Exact ties keep
the earliest candidate.

## Hold state machine

The hold variant enters its latched state only when both conditions are true
for two consecutive planning decisions:

1. the active reference is the final official subgoal; and
2. the current **real observation** satisfies the existing strict final-goal
   threshold `distance < 0.25 * reference_gap`.

Before confirmation it is identical to T1-L4. After confirmation it minimizes
the late-four mean and stays latched for the rest of the episode. It does not
read simulator truth, box count, placement count, reward, or success. Traces
record `hold_streak`, `hold_latched`, the retention rule, every candidate's
retention distance, and the selected cost.

This is a conservative first hold test. It intentionally does not add action
energy or latent-motion penalties, because those would introduce additional
weights and prevent attribution of the first result.

## Evaluation order

1. Run the persistent Colab screen on scene `100038`, sample seed `0`, full
   1795-step horizon. This is a debugging screen, not final evidence.
2. If a variant improves final retention without reducing peak count, freeze it
   before a paired multi-scene run.
3. Confirm on the existing development-scene list with seeds `0 2 3`. Do not
   change formulas after inspecting confirmation results.

Primary metrics are mean final boxes, official success, and four-box retention
`P(final >= 4 | peak >= 4)`. Secondary metrics are mean peak boxes, peak-to-final
loss, third-to-fourth conversion, and mean/P95 decision time. Comparisons are
paired by scene and seed and bootstrapped by scene when more than one scene is
available.

## Local verification

The deterministic tests cover the exact block/window definitions, reference-gap
normalization, stable ties, approach-versus-hold selection, two consecutive
real-observation confirmations, latch persistence, candidate budgets, call
counts, JSON traces, and previous A1–A5 behavior. They do not establish physical
task performance; that requires the GPU simulator run.

Completed on 2026-11-06 with Python 3.11 and CPU PyTorch:

```text
test_goal_*.py:       32/32 passed
test_onpolicy_a5.py:  10/10 passed
total:                42/42 passed
persistent notebook:  valid JSON; all 6 code cells passed AST parsing
git diff --check:     passed
```

The full physical screen was not executed locally because the workspace has no
CUDA device or downloaded 120k actor/WM artifacts. The persistent notebook is
the fixed GPU execution entry point and stores its outputs in Google Drive.
