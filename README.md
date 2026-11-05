> **For collaborators: DO NOT make changes directly on this branch. Create your own branch from `tianzi-branch/v2.0` and commit/push your work there. Keep this branch as the shared experiment reference.**

## What this branch works on

This branch finetunes a **goal-conditioned rectified-flow actor** for four-box stacking and uses a latent world model (WM) to select its proposed action plans. **A4-32 is this branch's best-performing tested configuration:** the **120k-step actor**, **32 candidate plans**, and the **original frozen WM**. On the fixed 20 development scenes × sampler seeds 0/2/3 (60 runs), it achieved **2.87 mean peak boxes, 2.13 mean final boxes, and 14/60 success (23.3%)**, with a mean decision time of **217 ms**. Its success rate ties A4-16; its mean peak and final box counts are higher. These are development results.

### What changed in the actor

The released actor conditions on the current observation latent. We add the **active goal latent** through a residual goal adapter. Current and goal latents use the same train-split normalization:

```text
current   = (z_current - mu_z) / sigma_z       # 192 dimensions
goal      = (z_goal    - mu_z) / sigma_z
m         = goal - current
p         = current * m                      # elementwise interaction
obs_token = original_obs_in(current) + goal_adapter(concat(m, p))
```

The adapter's final layer starts at zero; both the actor backbone and adapter are finetuned. Training goals are real future latents from the **same demonstration episode**, sampled at an offset `K` of 1–16 blocks. `K`, remaining time, and the next subgoal are not actor inputs. The original flow-matching velocity-MSE objective is retained. The image encoder/projector and WM remain frozen throughout A4.

### Training stages leading to A4-32

All counts below are **optimizer steps, not dataset epochs**. Each stage warm-starts from the preceding actor weights.

| Stage | Additional steps used | Cumulative steps | Change |
| --- | ---: | ---: | --- |
| 1. Goal conditioning | 50,000 | 50,000 | Initialize from the released actor, add the goal adapter; full 16-block training windows; LR `3e-5`. |
| 2. Continued goal training | 50,000 | 100,000 | Continue backbone + adapter training on full windows; LR `1e-4`. |
| 3. Tail coverage (A4) | 20,000 at the selected checkpoint | **120,000** | Include anchors near episode ends; sample `K` within the remaining valid length `R`; exclude padded action blocks from loss; LR `1e-4`. |

All three stages use effective batch 256 and EMA 0.999. The tail run was scheduled for an additional 50k steps, then stopped by the user; **A4 uses its step-020000 EMA**, explicitly selected at cumulative 120k. The original 150k target was not the checkpoint evaluated here.

### How the A4-32 planner selects candidates

1. Encode the real observation and official active subgoal with the frozen encoder. Advance the active goal only when the **real** squared latent distance is below `0.25 * ||g_active - g_previous||²`; the first previous reference is the official initial frame.
2. Sample **32 complete plans** from the goal-conditioned actor (`n_steps=10`, `lag=0`). Each plan has **16 blocks × 5 control steps × 25 action dimensions**, covering 8 seconds. Unnormalize and clip raw actions to `[-1, 1]`, then renormalize the actual bounded actions for the WM.
3. Roll out each plan for 16 future latents with the original frozen WM, using real observation/action history as context. Score each candidate by its **closest predicted approach to the active goal anywhere in the rollout**:

   ```text
   cost[n] = min over h=1..16 of ||z_imagined[n,h] - g_active||²
   chosen  = argmin over n=1..32 of cost[n]
   ```

   Exact ties keep the earliest candidate. Distances use the original, unnormalized latent coordinates.
4. Execute **only the chosen plan's first block** (0.5 seconds), observe again, and replan. Goal switching follows real observations; there is no next-goal lookahead or candidate reuse in A4-32.

### Key scripts and evidence

| Purpose | Repository path |
| --- | --- |
| Full design and experiment history | [notes/v2/DESIGN_2.0.md](notes/v2/DESIGN_2.0.md) |
| Actor, goal adapter, and future-goal/tail sampling | [assignment/training/training/goal_policy.py](assignment/training/training/goal_policy.py) |
| Actor training implementation | [assignment/training/training/train_goal_policy.py](assignment/training/training/train_goal_policy.py) |
| Training stage launchers | [Stage 1](scripts/train_goal_actor.sbatch), [Stage 2](scripts/train_goal_actor_round2.sbatch), [A4 tail coverage](scripts/train_goal_actor_a4.sbatch) |
| Planner sampling, scoring, and execution | [assignment/training/planning/goal_actor.py](assignment/training/planning/goal_actor.py) |
| Evaluation and per-decision evidence | [assignment/training/planning/evaluate_goal_actor.py](assignment/training/planning/evaluate_goal_actor.py) |
| Actual 120k A4-16/A4-32 evaluation launcher | [scripts/evaluate_goal_actor_a4_120k.sbatch](scripts/evaluate_goal_actor_a4_120k.sbatch) |
| A4 results, timing, and paired comparisons | [notes/v2/a4/evidence/summary.json](notes/v2/a4/evidence/summary.json), group `A4-32` |
| Exact evaluated models, hashes, scenes, seeds, and runtime | [notes/v2/a4/evidence/run.json](notes/v2/a4/evidence/run.json) |

The selected actor checkpoint on the cluster is:

```text
/mnt/projects/ll/tianzizh/cis6280-v2-a4-20261104/checkpoints/goal-actor-v2-a4-tail-seed0-20261104/step-020000.pt
SHA256: 755e95f0fac13fd8990436432986584ce0f8977a23eb5c8d946546fb55477ce3
```

The original frozen WM is `/mnt/projects/ll/tianzizh/cis6280-final-project/checkpoints/jepa/smwm-c3/model.pt`. Full A4-32 evidence is under:

```text
/mnt/projects/ll/tianzizh/cis6280-v2-a4-eval120k-20261104/
  outputs/planning/v2/a4-120k/
    run.json
    summary.json
    A4-32/episodes/scene-<scene>.sample-<seed>/
      episode.json
      steps.jsonl
      counts.jsonl
      video.mp4
```

All 60 runs have videos and step logs. The later A5 on-policy WM finetuning experiment did not improve the tested configuration; A4-32 continues to use the original WM. See [A5 results and evidence](notes/v2/a5/RESULTS.md).

---

# World Model Gym

This repository contains the MuJoCo dexterous manipulation simulator and the **CIS 6280 course assignment on four-box stacking with a latent world model**. A simulated 22-joint Sharpa Wave five-finger hand mounted on a 3-DoF wrist stage must grasp and stack four boxes (two 55 mm and two 85 mm tall) in a goal zone using only two wrist-camera RGB views.

<p align="center">
  <img src="docs/env_demo.gif" width="480" alt="Four-box stacking in VariedBoxEnv (front and side views)" />
</p>

## Course Assignment

Start with **[`assignment/README.md`](assignment/README.md)** for an overview of the task. The assignment offers two tracks:

- **[Planning Track (`assignment/planning/`)](assignment/planning/README.md)** — Keep our pretrained latent world model and action-chunk policy frozen, and design a test-time planner that scores candidate action chunks against visual subgoals. You can run and edit the entire baseline in Google Colab through [`assignment/planning/colab.ipynb`](assignment/planning/colab.ipynb) without local setup.
- **[Training Track (`assignment/training/`)](assignment/training/README.md)** — Build the five-stage pipeline from scratch, covering demonstration collection, world-model pretraining, diffusion-forcing policy training, on-policy predictor finetuning, and parallel closed-loop evaluation. Every stage has a pretrained artifact on Hugging Face so that you can start at whichever stage you want to explore.

## Local Installation

If you are running the training track or evaluating planners locally on Linux or macOS, create the Python environment and download the Sharpa Wave hand meshes with the setup script:

```bash
bash scripts/setup.sh
```

For headless GPU rendering on Linux servers, set `export MUJOCO_GL=egl` before running data collection or evaluation. Interactive viewer scripts and the full simulator specification are documented in:

- **[Simulator Guide (`docs/SIMULATOR.md`)](docs/SIMULATOR.md)** — Local installation, interactive macOS and Linux playback, observation and action spaces, and offscreen rendering.
- **Task and Data Specification ([`docs/SHARPA_BOX_STACKING.md`](docs/SHARPA_BOX_STACKING.md), [`docs/SHARPA_VARIED_BOXES.md`](docs/SHARPA_VARIED_BOXES.md))** — Scene generation, 25-DoF action semantics, support-graph tower scoring, and per-step contact labels.
