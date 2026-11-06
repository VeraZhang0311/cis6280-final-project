# Shafaa's planning-horizon experiment

Branch `shafaa` starts at Tianzi v2 commit `3638812`. This experiment uses the
released **original** actor and world model, with no model training. V2 goal
actor code remains inherited in the repository but is not used here.

Open `horizon_colab.ipynb` in Google Colab, select a T4 GPU, and Run all. This
self-contained notebook embeds our source; pushing the branch is not required.
The default is scene 100038 × sampler seed 0 × three horizons, full 1795-step
episodes with videos. It downloads original inputs without editing checkpoint
metadata. All three inputs are hashed before/after the experiment.

| Variant | Policy generation | WM prediction | Execute |
|---|---|---|---|
| 2 seconds | 16 blocks × 16 candidates | 4 blocks | 1 block |
| 4 seconds | 16 blocks × 16 candidates | 8 blocks | 1 block |
| 8 seconds | 16 blocks × 16 candidates | 16 blocks | 1 block |

Each block is 5 control steps / 0.5 seconds. Every variant keeps the official
subgoal sequence, tau=0.25, terminal squared latent L2 score, action handling,
and environment loop. In particular, candidate actions passed to the WM retain
the original baseline convention; raw executed actions are clipped as before.
The default prediction horizon equals `generate`, retaining the original path.
Policy generation always remains 16 so candidate sampling length is not a
second treatment. Earlier identical contexts/seeds share proposals; closed-loop
trajectories can diverge after the planner chooses different actions.

The 8-second variant should match the original planner with the same models,
hardware, dtype and seed. Tests check prefix length, selected candidate, actual
action context, real-only goal switching, and default/explicit-16 equivalence.
Compare the original frozen baseline directly for a further real-model check.

Run outside Colab on a CUDA machine from the repository root:

```bash
MUJOCO_GL=egl python -m assignment.planning.horizon_experiment \
  --policy checkpoints/policy/df-smwm-c1-sigma04/policy.pt \
  --world-model checkpoints/jepa/smwm-c3/model.pt \
  --goals dataset/goals/scripted-val100-every80/goals.npz \
  --scenes 100038 --sample-seeds 0 --planning-seconds 2 4 8 \
  --out outputs/planning/shafaa/horizons-demo
```

For implementation checks only, use `--episode-steps 40` and a separate output
folder. Short runs do not measure stacking performance. Expand later to a fixed
scene list and sampler seeds 0/2/3; compare paired scene/seed results. Primary
metric: mean peak boxes. Also report final boxes, success and mean/P95 decision
time. CUDA stage timings synchronize and exclude rendering/video writing.
Smaller WM horizons use less compute; this is not an equal-compute comparison.

`run.json` records inputs/source/settings/runtime. Each episode has `video.mp4`,
`decisions.jsonl`, and an atomic `episode.json` completion marker. Rerunning the
same configuration resumes completed episodes; changed settings require a fresh
output folder. `summary.json` is updated after every episode. Download the ZIP
from Colab to keep these files after the runtime expires.

Rebuild the embedded notebook after code changes:

```bash
python scripts/build_horizon_notebook.py
python -m unittest discover -s tests -p test_planning_horizon.py -v
```
