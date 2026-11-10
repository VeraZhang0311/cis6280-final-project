# v2.0 member-method integration

Branch: `tianzi/v2.0-integration`, based on v2.0 `050c748`. No member branch
is modified. Planner implementation is shared; `plan.json` selects the groups
and freezes the scenes, seeds, checkpoint hashes, and evaluation budget.

All groups use the exact released 120k EMA goal actor and original frozen WM.
The actor generates 16 blocks, uses 10 flow steps, and executes one block before
reobserving. Actions are clipped before WM prediction, and only real observations
advance goals. No model is trained or selected in this experiment.

| Group | Source and change |
| --- | --- |
| A4-32 / A4-16 | Fresh v2 baselines at both candidate budgets. |
| A4-16-RUIMING | Best reported `trajectory`: 16 candidates, full 16-block rollout, normalized minimum arrival/time score plus 0.25 endpoint distance; no lookahead or adaptive switching. |
| A4-32-T1 | Same whole-trajectory formula at 32 candidates, shared with Emoo's best screened T1. |
| A4-32-SHAFAA-2S/4S/8S | Endpoint scoring after 4/8/16 predicted blocks; always generate 16 blocks. Standardize to 32 candidates to match A4-32. |
| A4-32-T1-L4 | T1 retention term averages blocks 13 through 16. |
| A4-32-T1-H8 / H12 | T1 retention term uses block 8 / 12; full rollout and arrival search remain 16 blocks. |
| A4-32-T1-L4-HOLD | L4 until two consecutive real observations satisfy the final-goal threshold, then latch into late-four retention scoring. |

Ruiming's screenshot H=8 means eight seconds, corresponding to 16 half-second
blocks. Its time convention differs from a zero-indexed diagram only by a shared
constant, which cannot affect candidate ranking. His original normalization floor
is 1e-6; Emoo's is 1e-8, preserved separately. Other scoring weights are fixed at
0.05 for arrival time and 0.25 for retention. The worse Emoo T2, and Ruiming's
lookahead/adaptive variants, are not evaluated. Shafaa has no committed physical
results, so all three horizons are compared as an exploratory screen.

The 20 development scenes and policy seeds 0/2/3 exactly match A4-32: 60 episodes
per group, 11 groups, 660 episodes. Every episode saves video, per-decision traces,
counts and final metrics. All variants are paired against the freshly evaluated
A4-32; the 16-candidate method also compares against A4-16. Bootstrap resamples
scenes after averaging differences across their three seeds. Report peak boxes,
final boxes, official success, four-box retention and synchronized decision time.
This is development-set exploration with multiple comparisons, not independent
confirmation or a basis for claiming generalization.

Reuse the existing GRASP Python environment and runtime dependencies read-only.
Stage source and outputs into a new workspace. Run CPU unit checks on the login
node with single-threaded math; GPU smoke and physical evaluations use Slurm only.
Run `scripts/smoke_integration.sbatch` before `scripts/evaluate_integration.sbatch`.
The full evaluator validates the frozen plan and records its hash/content in
`run.json`. Checkpoint SHA256 must match the original A4 evidence. Fresh controls
allow a fair within-run comparison if the allocated GPU differs from the old L40.

Member source snapshots are pinned in `plan.json`; imported Emoo selector/tests
retain her follow-up definitions. `variants.py` is the shared mode registry.
