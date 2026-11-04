# v1.1 cluster run

## Current: G8/G9/G10 full search experiment (`v1.1-search-r4`)

**Final: job 1500169 COMPLETED, exit 0, elapsed 3h31m32s.** All 420 new
episodes are complete: 240 development (four configurations ×60) and 180
fixed-new-scene validation (three groups ×60). Another 120 G1/G7 development
reference cases are linked explicitly and are not new experiments. The final
report is `outputs/planning/v1.1/search-1500169/analysis/index.html`.
Complete evidence is currently being synchronized locally.

| Development group | Mean peak | Mean final | Peak >=2 | Peak >=3 | Mean / P95 decision ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| G8-32 | 1.800 | 1.467 | 37/60 | 16/60 | 156.4 / 193.2 |
| G8-96 | 1.850 | 1.550 | 35/60 | 18/60 | 342.9 / 391.4 |
| G9 | 1.783 | 1.333 | 38/60 | 11/60 | 293.6 / 328.4 |
| G10 | 0.550 | 0.417 | 2/60 | 0/60 | 432.1 / 466.8 |

G10-G8-96 paired peak difference is -1.300, 95% whole-scene bootstrap interval
[-1.567,-1.017], with 2 improved/17 tied/41 regressed episodes. This version of
policy-initialized iCEM substantially underperformed its 96-trajectory control.
The results do not identify whether model exploitation, the action projection
or other optimizer behavior caused the regression; no post-hoc tuning was done.

G8-96 won the predeclared development mean-peak selection; `selection.json`
locked it before any new-scene outcomes were evaluated.

| New fixed-scene validation | Mean peak | Mean final | Peak >=2 | Peak >=3 | Mean / P95 decision ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| G1 | 1.450 | 1.133 | 25/60 | 7/60 | 129.9 / 147.8 |
| G7 | 1.483 | 1.267 | 27/60 | 6/60 | 195.0 / 242.9 |
| G8-96 | 1.700 | 1.350 | 35/60 | 10/60 | 343.5 / 389.9 |

On the new scenes, G8-96-G7 mean peak difference is +0.217, 95% interval
[-0.083,0.500], with 25 improved/20 tied/15 regressed. G8-96-G1 is +0.250,
interval [-0.050,0.517]. Both include zero. Mean decision latency is about
76% higher than G7. G8-32/G9/G10 were not run on these validation scenes;
do not infer heldout superiority over those configurations.

The submission/progress notes below are historical snapshots.

Submitted job **1500169**, `cis6280-G8G10`, shared `batch`, constrained to
`dj-2080ti-0.grasp.maas` to match the completed G1/G7 reference host. Resources:
2 × 2080 Ti, four workers, eight CPUs, 32 GB RAM, six-hour limit. Liu's active
GPU nodes were fully allocated (the remaining Liu node drained), so shared
resources are used. Existing teacher job 1498746 remains untouched.

- Snapshot: `/mnt/projects/ll/tianzizh/cis6280-v11-search-r4`.
- Output: `/mnt/projects/ll/tianzizh/cis6280-v11-search-r4/outputs/planning/v1.1/search-1500169/`.
- Log: `/mnt/projects/ll/tianzizh/cis6280-v11-search-r4/outputs/v11-search-1500169.log`.
- Direct development full runs: G8-32 → G8-96 → G9 → G10, 60 each = 240 new episodes.
- Compatible complete G1/G7 from `sampling-1499337` are explicitly referenced,
  not silently pooled or relabelled. Actual GPU/dtype/dependency signatures are enforced.
- No pilot: direct full runs follow the user's stated preference. The design's
  planned pilot is not described as an executed experiment.
- The next 20 official scenes are already fixed in `validation-scenes-20-next.json`.
  After development, select the best new method using mean peak, mean final,
  lower mean decision time, then fixed configuration order; lock `selection.json`.
  G1, G7 and this method then run 60 episodes each on those new scenes, separately.
  Total new work is 420 episodes. Validation does not retune the method.

44 targeted tests passed on the cluster. Independent comparisons against the
immutable r3 source matched G0–G7 over 100 decisions per group (actions, RNG,
goals, costs and model calls). Development/validation report rendering passed
a synthetic smoke test including hyphenated G8 configuration names and galleries.
The old versioned G1–G7 evidence archive still verifies unchanged.

Every new episode keeps video, timeline, steps/events, six terminal keyframes
and tail contact evidence. Search metadata accounts for prefix/tail predictions,
all evaluated candidates, cached cost reuse, actual model batches and winner
source. Reports show development and fixed-scene validation separately.
Requeue/resume is automatic; partial episodes are archived, complete ones skipped.

```powershell
ssh tianzizh@grasp-login2.seas.upenn.edu "squeue -j 1500169; tail -20 /mnt/projects/ll/tianzizh/cis6280-v11-search-r4/outputs/v11-search-1500169.log"
scp -O -r tianzizh@grasp-login2.seas.upenn.edu:/mnt/projects/ll/tianzizh/cis6280-v11-search-r4/outputs/planning/v1.1/search-1500169 outputs/planning/v1.1/
```

Earlier completed protocols follow.

统一结果、两批可比范围和版本化证据入口见 [RESULTS.md](RESULTS.md)；本文件保留运行状态与历史沿革。

## Current: G6/G7 direct full run (`v1.1-sampling-r3`)

**Final: job 1499337 COMPLETED, exit 0, elapsed 1h30m37s.** G1/G6/G7 each
completed 60 cases (180 total). Final report is `analysis/index.html` under
`outputs/planning/v1.1/sampling-1499337`. All evidence is synchronized locally;
`analysis/local-evidence-check.json` verifies all 180 videos, logs, timelines,
tail snapshots and 1,080 keyframes. Final manifest status is `complete`.

| Same-allocation group | Mean peak | Mean final | Peak >=2 | Peak >=3 | Mean candidates | Mean / P95 decision ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | 1.467 | 1.150 | 24/60 | 12/60 | 16.00 | 126.6 / 145.4 |
| G6 | 1.533 | 1.233 | 30/60 | 10/60 | 23.97 | 196.5 / 388.7 |
| G7 | 1.717 | 1.433 | 35/60 | 10/60 | 48.00 | 197.0 / 246.3 |

G7's descriptive mean peak/final are highest; G1 reaches >=3 in more cases.
G7 versus G1 paired mean peak delta is +0.250, whole-scene bootstrap 95% CI
[-0.017, 0.533], with 26 improved/16 tied/18 regressed cases. G7-G6 is +0.183,
CI [-0.167, 0.517] (inverse of saved G6-G7). Both intervals include zero.

G7/G1 mean decision latency is 1.556x; G7/G6 is 1.003x. G6 therefore uses
roughly half G7's candidates without a meaningful observed average latency
saving; G7's P95 is 36.6% lower. G6 averages 1.498 policy/WM calls per decision
versus G7's single batched call. This is consistent with serial resampling
overhead/batch efficiency, not an isolated GPU profiler attribution.
G6 candidate distributions: 16 = 15,837 decisions, 32 = 678, 48 = 5,025;
26.5% trigger a second batch, 23.3% reach the third. Do not call the two methods
equal-compute simply because both have maximum candidate budget 48.

Launch/progress snapshots below are historical; actual final state is complete.

Job **1499337** is RUNNING on `dj-2080ti-0.grasp.maas`, shared `batch`:
2 × RTX 2080 Ti, four workers, eight CPUs, 24 GB host RAM, three-hour limit.
Liu's working GPU nodes are fully allocated; the third is drained. No shared
node had four available 2080 Ti GPUs, so this job uses two and reruns G1 in the
same allocation for a matched task/timing comparison. Teacher job 1498746 is untouched.

There is **no pilot**. Full sequence: G1 → G6 → G7, 60 episodes each, 180 new
episodes total. Historical G1 outcomes are not pooled. Old G0–G5 outputs remain
unchanged. G6 is adaptive 16/32/48 with the fixed 20% rule; G7 samples 48 once.
Both use G1 official goal switching and tail-min scoring, with no realignment.

- Snapshot: `/mnt/projects/ll/tianzizh/cis6280-v11-sampling-r3`.
- Output: `/mnt/projects/ll/tianzizh/cis6280-v11-sampling-r3/outputs/planning/v1.1/sampling-1499337/`.
- Log: `/mnt/projects/ll/tianzizh/cis6280-v11-sampling-r3/outputs/v11-sampling-1499337.log`.
- Reference cohort/input provenance: completed `bidir-1498961`; fresh G1 is the control.
- Completed validation: 34 targeted tests, 100-decision exact G0–G5 equivalence
  per group against r2, and a synthetic G1/G6/G7 report-rendering smoke test.

The current initial stage is full/G1. All new cases keep videos and the same
raw evidence package. Report generation and requeue/resume are automatic;
final `analysis/index.html` compares G6-G1, G7-G1 and G6-G7 task results and
sampling/latency costs. `run.json` contains actual progress and frozen parameters.

```powershell
ssh tianzizh@grasp-login2.seas.upenn.edu "squeue -j 1499337; tail -20 /mnt/projects/ll/tianzizh/cis6280-v11-sampling-r3/outputs/v11-sampling-1499337.log"
scp -O -r tianzizh@grasp-login2.seas.upenn.edu:/mnt/projects/ll/tianzizh/cis6280-v11-sampling-r3/outputs/planning/v1.1/sampling-1499337 outputs/planning/v1.1/
```

## Completed G0–G5 protocol

## Current: bidirectional G4 / combined G5 (`v1.1-bidirectional-r2`)

**Final: job 1498961 COMPLETED successfully (exit 0), elapsed 58m46s.**
All six full groups have 60 episodes each (360 total); six pilots add 72 episodes.
The final report is `analysis/index.html`, with videos and raw scalar evidence.
Final outputs are synchronized to the same local `bidir-1498961` folder.
Local verification confirms all 432 episode videos, scalar logs, timelines,
tail snapshots and 2,592 keyframe PNGs. The check is saved in
`analysis/local-evidence-check.json`.

| Full group | Mean peak | Mean final | Peak >=2 | Peak >=3 |
| --- | ---: | ---: | ---: | ---: |
| G0 | 1.467 | 1.283 | 29/60 | 8/60 |
| G1 | 1.650 | 1.383 | 34/60 | 9/60 |
| G2 | 1.350 | 1.150 | 24/60 | 8/60 |
| G3 | 1.533 | 1.100 | 30/60 | 9/60 |
| G4 | 1.667 | 1.233 | 34/60 | 11/60 |
| G5 | 1.567 | 1.267 | 30/60 | 13/60 |

G4-G0 paired mean peak delta is +0.200, 95% scene-bootstrap CI [-0.033, 0.433],
with 22 improved/24 tied/14 regressed episodes. G4-G3 is +0.133, CI [-0.083, 0.350].
G5-G4 is -0.100, CI [-0.433, 0.233]. All five treatment-vs-G0 mean-peak intervals
include zero. These descriptive rankings do not establish a reliable improvement.

The progress statements below are historical snapshots from launch/pilot completion.

New G4 pilot completed 12/12: mean peak 1.333, final 1.083, reached >=2 in
5/12 and >=3 in 0/12. Logs contain six forward and nine backward realignments,
with five direction reversals across consecutive realignments. This is a
mechanism observation, not a claim that backtracking improves task outcomes;
G3 pilot had mean peak 1.667/final 1.500. G5 is locked in the new `selection.json`
to G2 tail-weighted scoring + bidirectional alignment. G5 pilot completed 12/12:
mean peak/final both 1.083, >=2 in 5/12, >=3 in 0/12; three forward and seven
backward realignments. Both new pilots and their videos/reports are synchronized
locally under `outputs/planning/v1.1/bidir-1498961`.

The job is RUNNING on `kd-2080ti-1.grasp.maas`, now in full/G0, filling the 36
missing cases after reusing 24 completed compatible G0 cases. Total reusable
records: 72 (48 G0–G3 pilot + 24 full G0). Full evaluation is G0–G5 × 60 = 360
episodes; both new pilot groups were completed before entering this stage.

The revised DESIGN.md is implemented. New job **1498961** was submitted to
shared `batch` with 4 × 2080 Ti, eight workers, 16 CPUs, 48 GB and a two-hour
limit; requeue/automatic resume are enabled. Old resume job 1498825 was stopped
when the design changed. Existing teacher job 1498746 was left untouched.

- Source snapshot: `/mnt/projects/ll/tianzizh/cis6280-v11-bidir-r2`.
- New output: `/mnt/projects/ll/tianzizh/cis6280-v11-bidir-r2/outputs/planning/v1.1/bidir-1498961/`.
- Log: `/mnt/projects/ll/tianzizh/cis6280-v11-bidir-r2/outputs/v11-bidir-1498961.log`.
- New G4 = bidirectional matching + terminal scoring.
- New G5 = bidirectional matching + pilot-selected scoring.
- Reuse source: old `run-1498776`, completed compatible G0–G3 only. Old G4
  remains the historical single-direction combination and is excluded.

26 targeted tests passed on the cluster. An independent comparison with the
immutable old source matched G0–G3 exactly over 100 decisions each (actions,
RNG, goals, costs and model calls). Six-group full/pilot report generation also
passed a synthetic rendering smoke test.

Sequence: validate one real G4 pilot case → complete its 12 cases → lock G5
from the compatible fixed G1/G2 pilot → complete 12 G5 cases → full G0–G5,
60 cases each. Missing compatible baseline cases are filled; complete records
are linked with provenance and hashes. The same dtype/GPU/runtime is enforced.
The existing pilot ties on mean peak, so G2 is the expected scoring choice;
the new `selection.json` is written only after new G4 pilot completes.

```powershell
ssh tianzizh@grasp-login2.seas.upenn.edu "squeue -j 1498961; tail -20 /mnt/projects/ll/tianzizh/cis6280-v11-bidir-r2/outputs/v11-bidir-1498961.log"
scp -O -r tianzizh@grasp-login2.seas.upenn.edu:/mnt/projects/ll/tianzizh/cis6280-v11-bidir-r2/outputs/planning/v1.1/bidir-1498961 outputs/planning/v1.1/
```

`analysis/pilot.html` and `pilot-outcomes.png` are generated before full runs;
the final report is `analysis/index.html`. All new episodes retain videos,
timelines, scalar logs, contact/support tail evidence and terminal keyframes.

## Historical first protocol (old G4 naming)

Latest check: original job **1498776 was PREEMPTED** after 10 minutes by the shared
partition. G0–G3 pilot are complete (48 episodes); G4 pilot and full cohort were
not completed. Resume job **1498825** is submitted with `--resume`, same four
2080 Ti GPUs/eight workers, and currently PENDING (Priority). Requeue is enabled
for the resume submission. Output path remains `run-1498776`.

Pilot mean peak / final: G0 1.500 / 1.083; G1 1.333 / 0.917;
G2 1.333 / 0.917; G3 1.667 / 1.500. G4 is locked to G2 scoring because G1/G2
tie on the precommitted primary metric. G3 vs G0 has 3 improved, 8 tied and 1
regressed peak counts; scene-bootstrap 95% interval for mean peak difference
is [-0.167, 0.500]. This is pilot evidence from only four scenes.

All 48 completed pilot videos and raw evidence are synchronized locally.
Local `analysis/pilot-status.png` and `analysis/pilot-paired.json` summarize these
completed groups. The main `run.json` was left with status `running` by abrupt
Slurm preemption; actual scheduler state takes precedence until resumption.
Resume log: `/mnt/projects/ll/tianzizh/cis6280-v11-20261103/outputs/v11-resume-1498825.log`.

Submitted on 2026-11-03 from `tianzi-branch/v1.1` after all 19 targeted tests passed.
Implementation and fixed selection rules: [IMPLEMENTATION.md](IMPLEMENTATION.md).

- Original Slurm job: **1498776** (`cis6280-v11`; now PREEMPTED).
- Allocation: shared `batch`, `kd-2080ti-2.grasp.maas`, 4 × RTX 2080 Ti,
  8 workers, 16 CPUs, 48 GB host RAM, 3-hour limit.
- Liu was attempted first; job 1498774 remained pending with reason `Priority`
  and no estimated start, so only this newly submitted job was cancelled.
- The unrelated existing teacher job was left alone.
- Independent source snapshot: `/mnt/projects/ll/tianzizh/cis6280-v11-20261103`.
- Live outputs: `/mnt/projects/ll/tianzizh/cis6280-v11-20261103/outputs/planning/v1.1/run-1498776/`.
- Slurm log: `/mnt/projects/ll/tianzizh/cis6280-v11-20261103/outputs/v11-1498776.log`.
- Local initial manifest and test log: `outputs/planning/v1.1/run-1498776/`.

The pipeline is incomplete and awaiting resumption. It collects 12 pilot episodes per
group, locks G4, then 60 full episodes per group (360 total). Actual progress and
status live in `run.json`; completed episodes have `complete.json`.

All groups share this allocation and worker/video settings. New full G0 provides
the paired control because 2080 Ti inference uses FP16, while the historical B1
cohort used L40/BF16. Historical results are not pooled with this experiment.

At completion, `analysis/index.html` provides the visual report, videos and links
to raw evidence. `summary.json` stores the pilot/full statistics and scene-level
paired bootstrap intervals. `selection.json` stores the pilot-only G4 decision.

Status command:

```powershell
ssh tianzizh@grasp-login2.seas.upenn.edu "squeue -j 1498825; tail -20 /mnt/projects/ll/tianzizh/cis6280-v11-20261103/outputs/v11-resume-1498825.log"
```

Synchronize existing evidence to the local output folder (run from repository root):

```powershell
scp -O -r tianzizh@grasp-login2.seas.upenn.edu:/mnt/projects/ll/tianzizh/cis6280-v11-20261103/outputs/planning/v1.1/run-1498776 outputs/planning/v1.1/
```

Do not edit the remote source snapshot during the experiment. Resume after a
failure uses the same snapshot and `--resume`, identical worker/GPU settings and
output path; do not change parameters based on partial full-cohort outcomes.
