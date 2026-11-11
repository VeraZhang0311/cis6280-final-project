# A4-32-HOLD experiment

This experiment isolates the final-goal HOLD mechanism from the weaker T1/L4
approach score. Before the latch, `A4-32-HOLD` is exactly the released A4-32
selector: 32 candidates ranked by minimum active-goal distance over all 16
imagined blocks. It uses the same actor, world model, candidate budget, rollout,
goal switching, and one-block execution.

The latch activates after two consecutive real observations are within the
strict threshold for the final official goal. It remains active for the rest of
the episode. After latching, candidates are ranked by their mean distance over
imagined blocks 13--16, favoring sustained final-goal retention. Simulator box
counts, reward, and success are never read by the planner.

The follow-up `A4-32-LOCAL-HOLD` preserves the same pre-latch A4-32 behavior and
entry condition. While active, it considers a candidate safe only when all of
its first four imagined blocks remain within the final-goal threshold. Among
safe candidates it lexicographically minimizes first-block wrist XYZ RMS, then
full-action RMS, retaining candidate order as the final tie break. If no safe
candidate exists, it minimizes mean distance over those four blocks. Two
consecutive real observations outside the threshold exit HOLD and restore
A4-32. This rule was frozen after the original late-four HOLD retained four
boxes for only 0.5 seconds in the single-scene screen.

Run a fresh `A4-32` control and both HOLD variants on the frozen 20 development scenes
and sample seeds 0/2/3. Primary metrics are official success, mean final boxes,
and four-box retention. Secondary metrics are mean peak boxes, third-to-fourth
conversion, HOLD trigger count, false-positive HOLD outcomes, and decision time.
This is a paired development experiment, not independent confirmation.

## Frozen 20 x 3 launch

Stage this commit in a new GRASP workspace, then submit on the same L40 class as
the historical A4 evaluation:

```bash
sbatch --account=liu-account --partition=liu-compute --qos=ll-med \
  --gres=gpu:l40:1 \
  --export=ALL,TASK_ROOT=/mnt/projects/ll/tianzizh/cis6280-a4-local-hold-20261111 \
  scripts/evaluate_a4_32_hold.sbatch
```

The launcher validates the frozen plan and model hashes, evaluates 20 scenes x
3 policy seeds x 3 groups (180 episodes), supports exact resume, and writes the
automatic paired report under `outputs/planning/v2/a4-32-local-hold/`.
