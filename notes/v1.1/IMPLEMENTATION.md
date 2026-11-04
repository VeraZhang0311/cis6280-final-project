# v1.1 implementation and execution protocol

All G1–G7 experiments are complete. See [RESULTS.md](RESULTS.md) for the unified
results and the versioned evidence bundle; this file records implementation details.

## Current revision: `v1.1-sampling-r3` (G6/G7, no pilot)

G6 uses the unchanged G1 official goal selection and tail-min scoring. It samples
16 new candidates per batch, preserves all earlier candidates, and never updates
real history, executes actions or advances the goal between batches. At 16/32,
resample only when both `C_best >= T` and `(D_now-C_best)/max(D_now,1e-8) < .20`.
`D_now <= 1e-8` stops after the initial batch. Search ends at 48 unconditionally;
ties retain the earlier global candidate. Budget exhaustion is recorded as the
stop reason at 48 even if that last batch also achieves the improvement criterion.
The complete trace still records its achieved C_best/r.

G7 samples all 48 once, calls the WM once, and applies the same tail-min scoring.
Both execute only the first block of the global winner. G0–G5 fixed sampling is
unchanged. No goal realignment, warm start, privileged state or timeout advance
is enabled in either new group.

`v11_sampling` supports a frozen G1 reuse or `--fresh-reference`. The current
allocation has two 2080 Ti GPUs/four workers rather than the previous four/eight,
so it runs a fresh full G1, then full G6 and G7 in the same allocation. Each is
20 scenes × seeds 0,2,3, 60 episodes; no pilot or parameter selection is run.
Historical G1 outcomes remain separate and only supply fixed cohort/input provenance.
The selected G1 comes from prior development on these same scenes, not a new test set.

Logs include every sampled candidate cost, global selected index and batch,
per-batch best and cumulative best costs, relative improvement, extra-batch gain,
actual candidate count, policy/WM call counts, budget, thresholds and stop reason.
All videos, timelines, six keyframes and tail contact evidence remain enabled.
The automatic report compares task outcomes, mean/P95 decision time, 16/32/48
distributions, second/third batch rates, later-batch use and scene-paired outcomes.

34 targeted tests pass. An independent comparison with the completed r2 source
matches G0–G5 over 100 decisions per group (actions, RNG, goals, costs, model calls).
Tests specifically cover preserving old winners/ties, 16/32 early stops, the 48
cap under predicted deterioration, zero distances/thresholds, frozen context
within one decision, logging neutrality and the single G7 48-candidate batch.

Cluster paths and current job are in RUN.md. Prior protocols below are historical.

## Current revision: `v1.1-bidirectional-r2`

Updated DESIGN.md supersedes the original combined G4 definition below.
New G4 is **bidirectional real-observation alignment + terminal scoring**.
New G5 is **bidirectional alignment + pilot-selected G1/G2 scoring**.
The old G4 remains historical and is never relabelled as either new group.

G4/G5 search the clipped window `[m-3,m+3]`, including current m, after the
same 24-second dwell and 8-second stagnation gate. Match requires nonzero T
and q=D/T <1. Ties prefer current, then smaller absolute index change, then
earlier index. The same noncurrent best match must be observed three consecutive
times. Both forward and backward matches directly set the current goal for
that decision; official advancement is deferred until the next real observation.
History and pending confirmation reset after every change. Historical maximum
index is not a bound. There is no timeout-only advancement in G3/G4/G5.

New runner output uses `bidir-<jobid>` and records the design hash and explicit
G0–G5 definitions. Compatible completed G0–G3 pilot/full cases are validated
against original inputs, configs, worker settings and runtime, then linked with
original provenance and per-file hashes. Missing full cases are run normally.
Old G4 is excluded from reuse. G4's first pilot episode is audited before the
remaining eleven run. The same primary/tie rule selects G5's scoring from the
fixed G1/G2 pilot; G4 performance is not used to tune thresholds. G5 then runs
its own 12-episode pilot before all six 60-episode full groups are completed.

Logs distinguish nearby versus future matches, search mode, both directions
of realignment and normal advancement, matched q and confirmations. Summaries
include forward/backward counts and direction reversals (diagnostic proxies,
not proof of behavioral oscillation). Reports add G4-G3, G5-G4 and G5-selected-score
paired contrasts and a separate pilot gallery. All new episodes keep videos
and the original evidence package.

## Historical first implementation (superseded group naming)

Implemented on `tianzi-branch/v1.1`, based on `8d32302`. The design remains in
[DESIGN.md](DESIGN.md); this file records implementation decisions, not proven improvements.

- G0: original terminal squared L2 and official goal advancement.
- G1: minimum squared L2 over predicted blocks 13–16.
- G2: weighted tail distance `(D13 + 2 D14 + 3 D15 + 4 D16)/10`.
- G3: G0 scoring plus real-observation-only alignment. A goal must dwell 24 seconds;
  the inclusive last 8 seconds contain 17 observations at 0.5-second intervals.
  Compare medians of the first and last four distances. A decrease of at least 10%
  means continued improvement. Zero initial distance does not constitute positive improvement.
  Search at most three future goals, require `D/T < 1`, exclude zero thresholds,
  choose minimum ratio with earlier-index ties, and confirm the same goal three times.
  Official advancement runs first and again after alignment; every change clears history.
- G4: G3 with G1/G2 scoring selected using only the fixed 12-episode pilot. The
  primary selection metric is mean peak boxes. A difference no greater than one
  total box across 12 episodes counts as basically tied; ties prefer G2.
  This rule is fixed before running; `selection.json` locks its result.

All groups execute only the first block of the chosen candidate. No scoring change
adds model calls, consumes sampling RNG, uses simulator state, or alters goal files.
The existing parameters, 16 candidates, 16 predicted blocks, diffusion steps 10,
execute 1, lag 0 and tau 0.25, remain fixed.

`assignment.training.planning.v11` runs pilot G0 → G1 → G2 → G3 → G4, then full
G0 → G1 → G2 → G3 → G4 in one allocation with fixed worker and video settings.
Pilot scenes: 100038, 100069, 100078, 100152; seeds: 0, 2, 3.
Full scenes: the same 20 as the original B1 cohort. Pilot overlaps full and is not
an independent test set. Full G0 is freshly run because the new allocation can differ
from the old L40/BF16 baseline. Historical results are checked but never pooled as controls.
All 360 episodes are collected regardless of pilot rankings.

Each episode saves `summary.json`, `steps.jsonl`, `events.jsonl`, `video.mp4`,
`timeline.png`, six terminal keyframes and the existing 56-control-step tail evidence.
Logs include all candidate costs, the selected candidate's 16 predicted distances,
closest block over full horizon and tail, stagnation conditions, future-goal ratios,
confirmation count and actual goal-change reason. No full latent/action tensors are added.

New output layout is `outputs/planning/v1.1/run-<jobid>/{pilot,full}/G*/episodes/`.
The top-level manifest records source and input hashes, snapshot provenance,
settings, actual runtime, current stage and progress. Completed episodes have a
protocol-specific `complete.json`; `--resume` requires identical source and settings,
skips completed episodes, and archives partial episodes before rerunning them.

The offline `v11_report` builds `analysis/index.html`, outcome and paired-case plots,
alignment mechanism counts, all paired differences, censored transition observations,
and deterministic improved/tied/regressed example videos and prediction-distance plots.
Bootstrap intervals resample whole scenes, keeping all three sampler seeds together.
Latent distances and faster goal-index progress are mechanism evidence only.

Validation:

```text
python -m unittest discover -s tests -p 'test_v11*.py' -v
python -m unittest discover -s tests -p test_planning_diagnostics.py -v
```

Tests cover exact G0 actions/RNG/goal/model-call agreement with an independent legacy
step, independently constructed scoring cases, first-block execution, realignment
confirmations/reset/official advancement, pilot tie handling and scene-level pairing.
The Slurm script runs these checks before launching the complete pipeline.

Cluster submission and live paths are recorded separately in `RUN.md` after launch.
