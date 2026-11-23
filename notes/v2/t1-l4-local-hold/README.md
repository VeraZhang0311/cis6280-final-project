# T1-L4 HOLD optimization experiment

Status: corrected implementation and static checks complete; the corrected
single-scene screen and matched 20-scene x 3-seed L40 evaluation have not run.
The previous screen is superseded because it optimized a different pre-latch
selector. The pending cluster receipt is recorded in [launch.json](launch.json).

Branch: `emoo-a4-32-hold`. Base: `tianzi/v2.0-integration` at
`a8a12a4a34afa5306464763381da2dc0011b8a47`.

This experiment directly optimizes the released `A4-32-T1-L4-HOLD`. Before the
latch, every group uses the exact T1-L4 objective: normalized arrival distance
plus the arrival-time term and 0.25 times the mean normalized distance over
blocks 13--16. All groups use the same actor, world model, 32 candidates,
16-block rollout, real-observation goal switching, and one-block execution.

The latch activates after two consecutive real observations are within the
strict threshold for the final official goal and remains active for the episode.
Simulator box counts, reward, and success are never read by the planner.

Post-latch groups differ only as follows:

- `A4-32-T1-L4-HOLD`: minimize the mean distance over blocks 13--16.
- `A4-32-T1-L4-FIRST4-HOLD`: minimize the mean distance over blocks 1--4.
- `A4-32-T1-L4-LOCAL-HOLD`: require all first four blocks to be strictly inside
  the threshold, then lexicographically minimize wrist XYZ RMS, full-action RMS,
  and candidate order; if none is safe, minimize the first-four mean.

Run the three HOLD variants on the frozen 20 development scenes and sample seeds
0/2/3. Primary metrics are official success, mean final boxes,
and four-box retention. Secondary metrics are mean peak boxes, third-to-fourth
conversion, HOLD trigger count, false-positive HOLD outcomes, and decision time.
This is a paired development experiment, not independent confirmation.

## Frozen 20 x 3 launch

Stage this commit in a new GRASP workspace, then submit on the same L40 class as
the historical A4 evaluation:

```bash
sbatch --account=liu-account --partition=liu-compute --qos=ll-med \
  --gres=gpu:l40:1 \
  --export=ALL,TASK_ROOT=/mnt/projects/ll/tianzizh/cis6280-t1-l4-local-hold-20261123 \
  scripts/evaluate_t1_l4_local_hold.sbatch
```

The launcher validates the frozen plan and model hashes, evaluates 20 scenes x
3 policy seeds x 3 groups (180 episodes), supports exact resume, and writes the
automatic paired report under `outputs/planning/v2/t1-l4-local-hold/`.

Do not modify the selector, entry confirmations, latch behavior, local window, fallback,
scene list, seeds, models, or runtime after inspecting the screen. The full run
is a matched development confirmation, not an independent test set.
