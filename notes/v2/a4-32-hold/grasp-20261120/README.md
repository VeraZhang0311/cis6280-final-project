# Emoo HOLD evaluation on GRASP — November 20, 2026

**Completed: 180/180 episodes**, Slurm `1557533` exited `0:0` after 34:25.
See [full results](RESULTS.md). LOCAL-HOLD scored 11/60 successes versus 10/60
for fresh A4-32 and HOLD, but uncertainty includes zero. Its extra success over
HOLD occurred without HOLD ever activating, so it cannot be credited to the new
hold mechanism. A4-32 remains the default.

The `emoo` branch remains at `5c0bc8a`. New branch `emoo-a4-32-hold` adds six
commits after the integration snapshot `a8a12a4`:

| Commit | Change |
| --- | --- |
| `641ae17` | Isolate final-goal HOLD from the T1/L4 approach score. |
| `2796074` | Add a persistent Colab screen. |
| `257231e` | Add recoverable LOCAL-HOLD with a four-block safety window. |
| `551b890` | Freeze the matched 20-scene, three-seed evaluation. |
| `e8c0862` | Document the method and single-scene evidence. |
| `d4e185b` | Translate experiment records to English. |

All six commits were made November 10, 2026, America/New_York. The latest
implementation snapshot is `d4e185b86b52e05c37261d693ce235abe6def3e0`; its planner,
evaluator, variant registry, and tests match the frozen `257231e` implementation.
Only `tianzi/v2.0-integration` is updated for this evaluation. The member branch
is read-only.

## Frozen comparison

| Group | Candidates | Before HOLD | While HOLD is active |
| --- | ---: | --- | --- |
| A4-32 | 32 | Original full-trajectory minimum | Unchanged |
| A4-32-HOLD | 32 | Exactly A4-32 | Minimize mean distance over blocks 13–16; permanently latched |
| A4-32-LOCAL-HOLD | 32 | Exactly A4-32 | Require all first four predicted blocks inside the threshold; minimize first-block wrist RMS, then full-action RMS. If no candidate qualifies, minimize first-four mean distance. Exit after two real-observation misses. |

Both HOLD methods enter after two consecutive real observations inside the
strict final-goal threshold. They use no simulator box-count or success inputs.
The first three action dimensions are wrist target increments, so wrist RMS is
a command-magnitude proxy, not a direct measurement of actual wrist movement.

The original Colab evidence is one selected scene (`100038`), seed `0`:
baseline and long-horizon HOLD both finish with one box and fail; LOCAL-HOLD
finishes with four and succeeds. All three LOCAL-HOLD decisions used the
first-four-mean fallback. The small screen therefore does not validate safe
candidate filtering or wrist-RMS ranking. Full paired evaluation is needed.

The frozen plan in `../plan.json` is used unchanged: original 120k EMA actor,
original WM, official goals, the same 20 development scenes and seeds 0/2/3,
1795 control steps, 8 workers, and videos for all episodes. This is 60 episodes
per group, 180 total, on one L40 with bfloat16. Model/data hashes and dependency
versions match the November 10 integration evaluation. The fresh A4-32 control
will be checked against the 60 historical outcomes.

The first completed baseline episodes already show some differences from the
November 10 outcomes despite equal model/data hashes and package versions.
Initial observations and decisions match; later trajectories can diverge. The
cause is not established. Historical outcomes are therefore not reused as
controls. All comparisons use the fresh baseline in this same job. The audit
records historical and pre-HOLD replay differences rather than requiring bitwise
trajectory reproduction; selector logic is still covered by deterministic tests.

## Execution and validation

All 40 goal actor/policy tests and 10 A5 tests passed in the existing GRASP
runtime; see `preflight.json`. Existing environments and old experiment outputs
are reused read-only. Workspace:
`/mnt/projects/ll/tianzizh/cis6280-emoo-hold-eval-20261120-01`.

Smoke job `1557532` tests the three GPU interfaces. Full job `1557533` depends
on smoke success. Both use `liu-compute`, `ll-med`, and one L40; the full job
requests 16 CPUs and 48 GB. `launch.json` records the exact source and helper
hashes. At submission, no new full-batch performance results were available.

`run.sbatch` invokes Emoo's original `scripts/evaluate_a4_32_hold.sbatch`
unchanged, then runs the separate read-only `audit_hold.py`. The audit reports
HOLD entries/exits, safe-candidate and fallback use, selected wrist RMS, entries
before four boxes, and pre-HOLD agreement with baseline decisions. For
LOCAL-HOLD, the generic evaluator's `eligible` field denotes local safety;
its generic lookahead counters must not be interpreted as next-goal lookahead.

Full outputs are under `outputs/planning/v2/a4-32-local-hold/` in the new
workspace. These include all videos, step/count traces, automatic paired
`comparison.json` / `RESULTS.md`, and the separate `hold-audit.json`. The
source code, thresholds, candidate counts and evaluation plan are not retuned
after observing results. The scenes are reused development scenes, not an
independent held-out test.
