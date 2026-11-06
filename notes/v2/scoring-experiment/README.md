# A4-32 trajectory-scoring experiment

Date: 2026-11-06. This experiment changes only candidate selection while
holding the released 120k A4 goal actor, original frozen world model, official
goals, 32-candidate budget, 16-block rollout, action handling, goal switching,
and first-block execution fixed.

## Locked groups

| Group | Candidate cost |
| --- | --- |
| `A4-32` (R0) | Minimum active-goal squared distance over future blocks 1–16. |
| `A4-32-T1` (R1) | Normalized arrival distance + `0.05 * arrival/H + 0.25 * terminal_distance`. |
| `A4-32-T2` (R2) | Same arrival score with terminal weight increased to `0.50`. |

Distances for T1/T2 are normalized by the active reference gap
`||g_active - g_previous||^2`. The arrival block minimizes normalized distance
plus the time term. Exact cost ties retain the earliest candidate. T1 copies
the independently screened PARCC trajectory weights. T2 is a predeclared test
of whether stronger terminal retention converts transient peak progress into
better final count or official success.

There is no next-goal lookahead, adaptive goal advancement, candidate reuse,
or action-space optimization. Those mechanisms did not show consistent gains
in prior experiments, and adding them would prevent attribution to scoring.

## Run

Open `cis6280_planning_persistent.ipynb` from the `emoo` branch and run all
cells on a Colab GPU. The notebook downloads and verifies the released actor,
original WM, official actor, and official dense goals; then runs all three
groups for validation scene `100038`, sample seed `0`, and the full 1795-step
horizon. Outputs persist under:

```text
MyDrive/cis6280-final-project-v2/outputs/a4-32-scoring/<UTC timestamp>/
```

For a multi-scene confirmation run, invoke the same evaluator with a frozen
scene file and seeds `0 2 3`:

```bash
python -m assignment.training.planning.evaluate_goal_actor \
  --policy checkpoints/goal-actor-a4-120k-ema.pt \
  --a4-cumulative-step 120000 \
  --world-model checkpoints/jepa/smwm-c3/model.pt \
  --official-policy checkpoints/policy/df-smwm-c1-sigma04/policy.pt \
  --goals dataset/goals/scripted-val100-every80/goals.npz \
  --scene-file notes/v2/a5/reserved-scenes.json --sample-seeds 0 2 3 \
  --groups A4-32 A4-32-T1 A4-32-T2 \
  --horizon 1795 --workers 1 --gpus 0 --videos 20 \
  --out outputs/planning/v2/a4-32-scoring-confirm
```

Do not tune weights using the reserved-scene result. Report mean peak, mean
final, official success, third-to-fourth conversion, decision time, and the
scene-grouped paired intervals emitted by the evaluator.

