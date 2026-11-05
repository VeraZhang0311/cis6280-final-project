# v2 goal actor inference

`GoalActorAgent` in [goal_actor.py](goal_actor.py) loads the explicit
`goal-df-policy-v2` checkpoint and freezes both actor and world model.
All three modes use the same locked checkpoint and the official subgoal images.

| Mode | Candidates | Selection | Predictor calls per decision |
| --- | ---: | --- | ---: |
| A1 | 1 | Execute the sampled plan's first block | 0 |
| A2 | 16 | Minimum active-goal distance across future blocks 1–16 | 16 batched steps |
| A3 | 16 | Active-first eligibility and next-goal lookahead | 16 batched steps |

Each decision generates 16 blocks with `n_steps=10, lag=0`, executes exactly
one block of five control steps, then reads a new observation and replans.
The actor receives `[B,1,192]` current and active-goal latents, normalized using
the checkpoint's shared train statistics. K, remaining time and next goal are
never actor inputs. WM context and distances use unnormalized frozen latents.

Only the current **real** observation can advance the active goal, monotonically,
with strict `distance < 0.25 * reference_gap`. The first reference gap uses the
official init image; later gaps use consecutive reference goals. There is no
timeout, goal rollback or advancement based on imagined states.

A3 computes `q = min(D_active[:, :15]) / max(T_active, 1e-8)` and
`r = D_next[:, 16] / max(T_next, 1e-8)` from the same rollout. It considers only
`q < 1` candidates and minimizes `q + 0.25*r`. If none qualify, or the active
goal is the last goal, it falls back to A2 over all 16 future blocks. Both modes
keep the earliest sampled candidate on exact cost ties. The switch tau and
lookahead weight are separate fixed constants despite both equaling 0.25.

The simulator action interface remains raw actions clipped to [-1,1]. Candidates
are unnormalized and clipped before WM scoring, then normalized again; the WM
therefore predicts the actions that would be executed. The rolling WM context
contains only real observations and the actual clipped first blocks previously
executed. It does not retain the other 15 imagined action blocks.

## Run and evidence

```bash
python -m assignment.training.planning.evaluate_goal_actor \
  --policy checkpoints/goal-actor-run/policy.pt \
  --world-model checkpoints/jepa/smwm-c3/model.pt \
  --goals dataset/goals/scripted-val100-every80/goals.npz \
  --scenes 100038 --sample-seeds 0 --groups A1 A2 A3 \
  --horizon 1795 --workers 1 --gpus 0 --videos 1 \
  --out outputs/planning/v2/goal-actor-comparison
```

`--scene-file` also accepts a fixed JSON list of scene seeds, or an object with
`scene_seeds`/`scenes`. Scene objects may use `scene_seed`. The evaluator checks
the requested existing split (default val), goal availability, duplicate seeds,
and checkpoint type before launching workers. Freeze a new formal validation
scene list separately as required by the design. Default sample seeds are
0, 2, 3. `--gpus` indexes devices within inherited `CUDA_VISIBLE_DEVICES`, so
Slurm allocations are respected. `--videos N` records the first N scenes for
each group and sample seed. `--official-policy` optionally records the original
actor hash for provenance. Use a new output directory for each evaluation.

After interruption, pass `--resume` with the same configuration and source.
The evaluator verifies model/goal hashes, scene/seed lists, settings and source
hashes against `run.json`. It reuses episodes with an atomic `episode.json`
completion marker and their finished evidence, and reruns partial episodes
from the scene's initial state with the same sampler seed. `attempts.jsonl`
records Slurm restarts. Summary `wall_s` describes the current attempt; decision
timings remain those measured when each episode actually ran.

```text
<out>/run.json                       # model/goal SHA256, code, scenes, seeds, fixed settings
<out>/summary.json                   # group statistics and scene-paired bootstrap differences
<out>/A1/summary.json
<out>/A1/episodes/scene-100038.sample-0/
  episode.json                      # official counts/success, first-reach times, call totals
  counts.jsonl                      # actual count at every control step
  steps.jsonl                       # one record per observed decision / executed block
  video.mp4                         # observed image, active goal, count/peak; A3 also next goal
<out>/A2/...
<out>/A3/...
```

Decision traces contain active index before/after switching, real distance and
threshold, switch events, chosen candidate, every candidate's minimum distance
and nearest future block (1-based), A3 q/r/eligibility/selection cost and fallback,
component/total timing and call counts. Ineligible A3 selection costs are JSON
null, not Infinity. Full candidate action/latent tensors are not saved.

Call counts distinguish policy sampling (one call) from its ten velocity
forwards. One WM rollout has 16 batched predictor steps; A3 adds no second
rollout. Goal/init encoding at reset is recorded separately from per-decision
encoder calls. CUDA timings synchronize around measured stages and exclude
rendering, reset and video writing.

Reports include unconditional peak/final reach rates, mean peak/final count,
official task success and pooled mean/P95 decision time. First-to-second-box
records retain all episodes reaching the first box, with censoring at the end
when a second box is not reached. The summary reports conversion, number
censored and Kaplan–Meier median time when estimable. Paired A2−A1/A3−A2
bootstrap intervals resample scenes, keeping a scene's sample seeds together.
No interval is claimed from a single scene.

The existing `evaluate --kind A1|A2|A3` and `scripts/evaluate_stack_goal_actor.py`
entry points are supported. Use the new evaluator for the structured evidence
above. The legacy `--kind goal` retains its original final-latent score and is
not the design's fair A0 baseline; A0 is outside the A1–A3 implementation.

## Verification

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_goal_actor.py -v
```

Tests cover full-trajectory minima, normalized lookahead, eligibility at blocks
1–15, fallback including block 16, stable ties, zero thresholds, real-only goal
updates, A1's zero predictor calls, active-only actor inputs, clipped executed
action history, reseeding, censoring, scene-paired aggregation and video panels.
Use a short horizon for implementation checks; such runs do not measure task
performance or select a training checkpoint.

## A4-16 and A4-32

These modes use the tail-coverage actor trained from the locked 100k parent.
The original final target was 150k. The user stopped training and selected the
cumulative 120k `step-020000.pt` EMA; pass `--a4-cumulative-step 120000` for that
evaluation. The default gate still expects the original 150k target. Tail coverage,
local/cumulative steps and parent SHA256 must match the explicit selection.

A4-16 samples 16 candidates; A4-32 samples 32 candidates in one policy call.
They otherwise use exactly A2's 16-step full-trajectory minimum, stable tie
break, clipping/normalization, real-observation goal switching, and first-block
execution. They use no next-goal lookahead or candidate reuse. Their modes are
available in both evaluation entry points; defaults remain A1/A2/A3.

[evaluate_goal_actor_a4.sbatch](../../../scripts/evaluate_goal_actor_a4.sbatch)
fixes 20 development scenes x sample seeds 0/2/3, 8 workers on a Liu L40,
1795 control steps, and all videos for A4-16/A4-32 (60 each). Prepare the same
fixed scene list as `inputs/scenes-20.json`. The existing 100k A2 launch receipt
(`outputs/planning/v2/round2-final-20261104/launch.json` in the local project)
goes in `inputs/A2-launch.json`; it supplies GPU provenance missing from the
older evaluation manifest. This launcher assumes the source snapshot is in its
separate A4 workspace and the locked 150k actor has finished training there.

`--reference-run <completed-100k-evaluation> --reference-launch <launch.json>`
reuses the old A2 episodes only after matching the parent actor hash, scene/seed
coverage, WM/goals hashes, task, scoring/action handling, GPU, dtype, library
versions, workers and video settings. If these differ, rerun A2 with the old
100k checkpoint under matched conditions. The reference remains read-only and
its manifest, launch and episode hashes are recorded. Reports include
scene-paired A4-16−A2 and A4-32−A4-16 comparisons.

Every episode and group report now includes third-to-fourth-box conversion,
event/censor counts and Kaplan–Meier median time. This is derived from the
saved first-reach times for old A2 episodes, without altering their evidence.
A4-16−A2 combines the sampling change and 20k extra training in the current run; it is not an
isolated causal estimate of tail coverage.

## A5-16 and A5-32

These modes freeze the same 120k actor and use the shared on-policy adapted WM.
Pass `--wm-selection <wm-a5/selection.json>` with the selected WM. They reuse
A4's candidate counts, full-trajectory scoring, action handling and goal switching.
`--reference-run <completed-a4-120k>` verifies matched conditions and reuses both
same-budget A4 groups for scene-paired comparisons. Full data, training,
W&B, checkpoint selection and recovery instructions are in [A5.md](../training/A5.md).
