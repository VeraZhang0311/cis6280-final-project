# GRASP HOLD / LOCAL-HOLD batch results

Completed November 20, 2026. Slurm job `1557533`: `COMPLETED`, exit `0:0`, elapsed `00:34:25`. All 180 episodes and the post-run mechanism audit completed successfully.

**The batch does not establish a reliable improvement over A4-32, or an advantage of LOCAL-HOLD over ordinary HOLD.** LOCAL-HOLD has a small positive point estimate, but its confidence intervals include or touch zero. Its apparent one-success advantage over HOLD occurred in a case where neither method entered HOLD, so it cannot be attributed to the new hold mechanism.

## Fixed setup

- Frozen member source `d4e185b86b52e05c37261d693ce235abe6def3e0`, identical planner/evaluator/registry/tests to the frozen `257231e` implementation. No selector, model, threshold, candidate count or evaluation setting was retuned.
- Original 120k EMA actor, original frozen WM and official goals, verified by SHA256. The same 20 development scenes and sample seeds 0/2/3 were used: 60 episodes per group, 180 total.
- One NVIDIA L40, bfloat16, eight workers, 1795-control-step horizon, 32 candidates, 16 generated/imagined blocks, one executed block. All episodes have videos and step/count traces.
- The existing GRASP runtime was reused read-only. All 40 goal actor/policy tests and 10 A5 regression tests passed. Member branches were not modified.

## Full results

| Variant | Candidates | Mean Peak Boxes | Mean Final Boxes | Success Rate | Four-box Retention | Mean / P95 Decision (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A4-32 | 32 | 2.817 | 2.067 | 10/60 (16.7%) | 10/15 | 212.0 / 237.3 |
| A4-32-HOLD | 32 | 2.817 | 2.067 | 10/60 (16.7%) | 10/15 | 212.5 / 238.2 |
| A4-32-LOCAL-HOLD | 32 | 2.850 | 2.100 | 11/60 (18.3%) | 11/16 | 212.5 / 237.4 |

## Paired uncertainty

Differences below are variant minus the fresh A4-32 control. Bootstrap resamples scenes after averaging their three seed differences (20 scenes, 2000 draws). These are exploratory, unadjusted 95% intervals on reused development scenes.

| Variant | Peak Delta [95% CI] | Final Delta [95% CI] | Success Delta [95% CI], pp |
| --- | ---: | ---: | ---: |
| A4-32-HOLD | +0.000 [-0.050, +0.050] | +0.000 [-0.150, +0.183] | +0.000 [-5.000, +5.000] |
| A4-32-LOCAL-HOLD | +0.033 [+0.000, +0.083] | +0.033 [-0.100, +0.217] | +1.667 [+0.000, +5.000] |

## Mechanism audit

| Variant | Triggered Episodes | Entries / Exits | Latched Decisions | Safe-ranking Decisions | Fallback Decisions | Entries Below Four Boxes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A4-32 | 0/60 | 0 / 0 | 0 | 0 | 0 | 0 |
| A4-32-HOLD | 13/60 | 13 / 0 | 431 | 0 | 0 | 12 |
| A4-32-LOCAL-HOLD | 13/60 | 13 / 8 | 37 | 2 | 35 | 12 |

LOCAL-HOLD used its safe-candidate/wrist-ranking branch in only **2/37 latched decisions (5.4%)**; the other **35/37 (94.6%)** used the first-four-mean fallback. Safe-ranking occurred in scene 100072/seed 2 and scene 100186/seed 0; their task outcomes did not improve relative to the fresh baseline. This does not validate wrist ranking as the source of any gain.

Both HOLD variants triggered in 13 episodes, six of which ultimately failed. Twelve of 13 entries occurred while the real box count was below four. These are post-hoc diagnostics only; simulator counts never entered planner decisions. Visual final-goal proximity must not be interpreted as verified task completion.

### Where outcome differences came from

For every episode in which either HOLD variant entered HOLD, their peak/final/success outcomes were identical. LOCAL-HOLD versus HOLD differed in only two episodes, both with zero entries for both variants:

| Scene | Seed | HOLD Peak / Final / Success | LOCAL-HOLD Peak / Final / Success | HOLD Entries (Both) |
| --- | ---: | --- | --- | ---: |
| 100060 | 0 | 3 / 2 / Fail | 4 / 4 / Success | 0 |
| 100098 | 3 | 1 / 1 / Fail | 2 / 1 / Fail | 0 |

The extra LOCAL-HOLD success over HOLD therefore reflects an untriggered replay difference, not an exercised local selector or recovery rule. Against A4-32, both HOLD variants did convert scene 100169/seed 3 from failure (peak/final 3/0) to success (4/4); both also worsened final counts in other triggered cases. That one shared success case supports further investigation of final-goal behavior, not a broad benefit or a unique LOCAL-HOLD advantage.

## Repeatability and screen reproduction

Fresh A4-32 reproduced only **25/60** historical November 10 peak/final/success outcome triples, despite equal model/data hashes and package versions. Initial observations and decisions matched in the inspected cases, while later trajectories diverged. The cause was not established. The historical 14/60 success must not be substituted for this job's fresh 10/60 control.

Within this job, each HOLD variant had two episodes with pre-HOLD candidate/index replay differences. These were recorded rather than discarded. Inferences about small aggregate gains must account for the observed repeatability limits.

The selected Colab case (scene 100038, seed 0) did not reproduce its LOCAL-HOLD success on L40/BF16:

| Variant | Peak Boxes | Final Boxes | Success | HOLD Entries / Exits | Safe / Fallback Decisions |
| --- | ---: | ---: | --- | --- | --- |
| A4-32 | 4 | 3 | No | 0 / 0 | 0 / 0 |
| A4-32-HOLD | 4 | 1 | No | 1 / 0 | Not applicable |
| A4-32-LOCAL-HOLD | 4 | 1 | No | 1 / 1 | 0 / 2 |

The earlier Colab summary records LOCAL-HOLD peak/final 4/4 and official success. Its full runtime manifest was not part of the committed screen evidence, so an exact hardware/dtype comparison is unavailable. No algorithm changes were made to recover that selected case.

## Evidence

- [Full comparison and all paired intervals](evidence/comparison.json), [run manifest](evidence/run.json), [aggregate summary](evidence/summary.json).
- [HOLD audit, including all 180 episode summaries](evidence/hold-audit.json), [outcome attribution](evidence/attribution.json), [independent recomputation](evidence/validation.json).
- [SHA256 file index](evidence/file-index.json), [portable evidence bundle identity](evidence/bundle.json), [actual scheduler launch receipt](launch.json).
- Local complete metrics bundle: `outputs/emoo-hold-eval-20261120/emoo-hold-evidence-180.tar.gz`; extracted 180 episode records: `outputs/emoo-hold-eval-20261120/evidence/`.
- GRASP videos and full traces: `/mnt/projects/ll/tianzizh/cis6280-emoo-hold-eval-20261120-01/outputs/planning/v2/a4-32-local-hold/`.

All 196 downloaded evidence files were independently verified against their recorded SHA256 values. No new held-out evaluation or retraining was performed. A4-32 remains the default; this batch does not justify promoting LOCAL-HOLD.
