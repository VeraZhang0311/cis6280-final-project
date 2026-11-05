# Planning Track

Local PARCC experiments, the trajectory planner, ablations, and paired statistical
analysis are documented in [EXPERIMENTS.md](EXPERIMENTS.md).
Completed GPU screening results are in [RESULTS.md](RESULTS.md).

## Group Planning Quick Start

Run commands below from the repository root. Python 3.12 and an NVIDIA GPU with
headless EGL rendering are used for the experiments. Setup downloads the frozen
models, visual subgoals, and hand meshes; no model training is required.

### Install and Choose Artifact Paths

```bash
git clone --branch planning/trajectory-ablation-results https://github.com/VeraZhang0311/cis6280-final-project.git
cd cis6280-final-project

# Change these to writable large-file storage on your machine.
export PLANNING_MODEL_ROOT="$PWD/checkpoints/planning"
export PLANNING_DATA_ROOT="$PWD/datasets/planning"
bash scripts/setup_planning.sh
```

On PARCC, the already downloaded artifacts can be reused instead:

```bash
export PLANNING_MODEL_ROOT=/vast/projects/xuhw/healthworldx/zhangrm/model/cis6280-stack-world-models-v2
export PLANNING_DATA_ROOT=/vast/projects/xuhw/healthworldx/zhangrm/data/cis6280-stack-datasets-v2
```

Keep those variables set when submitting jobs. The setup script pins PyTorch
2.9.1 and MuJoCo 3.13.0 and downloads the exact Hugging Face revisions recorded in
[EXPERIMENTS.md](EXPERIMENTS.md). Model weights, datasets, and new run outputs stay
outside Git.

### Available Configurations

| Name | Candidate generation and selection |
| --- | --- |
| `policy` | One 8-second policy sample, without goal scoring or predictive rollout. |
| `baseline` | 16 candidates; minimize predicted final latent distance to the current subgoal. |
| `clipbaseline` | Baseline scoring, with actions clipped before predictive rollout. |
| `trajectory` | Clip actions; score intermediate arrival, arrival time, and terminal distance. |
| `lookahead` | Trajectory scoring plus a bounded, ordered next-subgoal bonus. |
| `adaptive` | Trajectory scoring plus conditional subgoal advancement after observed stagnation. |
| `combined` | Trajectory scoring, next-subgoal bonus, and stagnation recovery. |
| `budget32` | Baseline scoring with 32 candidates. |

Every configuration generates 16 action blocks (8 simulated seconds) and executes
one block (0.5 seconds) before generating again. Each block contains five 0.1-second
control actions. All model weights stay frozen. The `trajectory` configuration
explicitly disables lookahead and recovery; constructing `TrajectoryPlanningAgent`
without these overrides enables both, corresponding to `combined`.

### Run a Small Check

The runner prints commands and records a prepared manifest unless `--run` is
supplied. This first command does not launch evaluation:

```bash
.venv/bin/python -m assignment.planning.experiments --stage smoke --variants baseline trajectory
```

Inside an allocated GPU session, evaluate two scenes with policy seed 0:

```bash
MUJOCO_GL=egl .venv/bin/python -m assignment.planning.experiments \
  --stage smoke --variants baseline trajectory --workers 2 --run
```

Or submit from the repository root on PARCC; replace the account if necessary:

```bash
mkdir -p runs/planning
sbatch --account=xuhw-healthworldx --partition=dgx-b200 --qos=dgx \
  --gres=gpu:1 --cpus-per-task=28 --mem=224G --time=00:30:00 \
  --output='runs/planning/smoke-%j.log' \
  scripts/planning.slurm smoke baseline trajectory
```

Reports appear under `runs/planning/smoke/<variant>/`; smoke also saves videos.
Use `squeue -u "$USER"` to check jobs. Once all smoke reports are complete:

```bash
.venv/bin/python -m assignment.planning.summarize \
  --stage smoke --variants baseline trajectory
```

### Reproduce the Eight-Method Screen

This submits 24 tasks, each covering one method and ten scenes with three policy
seeds. At most two GPUs run concurrently. Keep the variant order fixed across
tasks because it determines the array-index mapping.

```bash
mkdir -p runs/planning
sbatch --account=xuhw-healthworldx --partition=dgx-b200 --qos=dgx \
  --gres=gpu:1 --cpus-per-task=28 --mem=224G --time=00:30:00 \
  --array=0-23%2 --output='runs/planning/screen-%A_%a.log' \
  scripts/planning.slurm screen baseline combined policy clipbaseline trajectory lookahead adaptive budget32

# Run after all 24 tasks complete.
.venv/bin/python -m assignment.planning.summarize --stage screen
```

For one method in an allocated GPU session, use the experiment runner with
`--stage screen --variants trajectory --run` instead. Stage `confirm` uses the
remaining 270 validation scenes; it has not been run. See
[EXPERIMENTS.md](EXPERIMENTS.md) for the confirmation protocol and array commands.

### Inspect Existing Results Without a GPU

The completed screen is committed under `results/planning/2026-11-05-screen/`,
including all 72 JSON reports containing 720 episode records. Recompute statistics:

```bash
.venv/bin/python -m assignment.planning.summarize \
  --root results/planning/2026-11-05-screen --stage screen
.venv/bin/python -m unittest discover -s tests -p 'test_planning*.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_assignment.py' -v
```

Summarization uses only the Python standard library and can also run with
`python3`, without downloading models or installing the GPU environment.
Results are screening estimates, not held-out test performance; the observed best
method's improvement is not statistically established. See [RESULTS.md](RESULTS.md).

## Original Course Planning Guide

In this track you keep our pretrained world model and policy frozen and write a better test-time planner. For each validation scene you get a sequence of subgoal images, one every 8 seconds of a scripted reference run. Your planner decides which actions to take and when to move on to the next subgoal.

<p align="center">
  <img src="../../docs/planning_demo.gif" width="384" alt="GoalPlanningAgent rollout (top: live front and side views; bottom: active subgoal)" />
</p>

## Pretrained Models and Subgoals

The Colab notebook downloads three files from Hugging Face ([models](https://huggingface.co/YongYong/cis6280-stack-world-models-v2), [datasets](https://huggingface.co/datasets/YongYong/cis6280-stack-datasets-v2)):

1. **World model** (`jepa/smwm-c3/model.pt`) — Encodes each camera frame into a 192-D latent and predicts how that latent will change under a sequence of actions. See the [training track](../training/README.md) for how it was trained.
2. **Policy** (`policy/df-smwm-c1-sigma04/policy.pt`) — A [Diffusion Forcing](https://arxiv.org/abs/2407.01392) policy that proposes the next 16 action blocks (8 seconds of motion) from the current frame.
3. **Subgoal images** (`goals/scripted-val100-every80/goals.npz`) — Frames from a scripted run of the same scene, one every 8 seconds. The full benchmark uses the 300-scene version, `goals/scripted-val300-every80/goals.npz`.

## Quick Start (Google Colab)

Open [`colab.ipynb`](colab.ipynb) in [Google Colab](https://colab.research.google.com/), switch the runtime to a GPU (**Runtime → Change runtime type → GPU**), and click **Runtime → Run all**. In about 15 minutes, the notebook:

1. Installs MuJoCo and downloads the code, hand assets, models, and subgoals.
2. Shows the subgoal images for validation scene `100038`.
3. Defines `GoalPlanningAgent` in a code cell that you can edit.
4. Runs the scene with and without planning and shows both videos.

## How the Baseline Planner Works

The baseline is `GoalPlanningAgent` in [`assignment/training/planning/agent.py`](../training/planning/agent.py), with the same code in `colab.ipynb`. Every 5 control steps (0.5 s), it encodes the current camera frame into a latent $z$ and then:

1. **Picks the active subgoal.** It moves on from subgoal $g_m$ once the current latent is close to it, relative to the distance between $g_m$ and the previous subgoal (for the first subgoal, the episode's first frame):
   $$\|z - g_m\|_2^2 < \tau \, \|g_m - g_{m-1}\|_2^2, \qquad \tau = 0.25.$$
2. **Samples candidates.** The policy proposes 16 candidate action chunks, each 16 blocks (8 s) long.
3. **Imagines and selects.** The world model rolls every candidate forward from the last few frames, and the planner keeps the candidate whose predicted final latent is closest to the subgoal:
   $$n^\star = \arg\min_n \big\|\hat{z}^{(n)}_{\text{final}} - g_m\big\|_2^2.$$
   It executes only the first block of that chunk (5 control steps) and then replans.

### Designing Your Own Planner

To try your own planner for the competition, keep the three downloaded files and the environment loop fixed, and modify `GoalPlanningAgent` in `colab.ipynb` or in [`assignment/training/planning/agent.py`](../training/planning/agent.py). Your agent needs two methods:

- `reset(goal_frames, init_frame, seed=None)` — Called at the start of each episode with the subgoal images (`(M, 128, 256, 3)`, `uint8`), the first frame (`(128, 256, 3)`, `uint8`), and an episode seed.
- `step(frame)` — Called every 5 control steps with the current frame; returns the next 5 actions as a `(5, 25)` `float32` array in `[-1, 1]`.

Some ideas: score the whole predicted trajectory instead of only its last latent, look ahead to several upcoming subgoals, refine candidates iteratively (for example with CEM or MPPI), change how many candidates you sample or how many blocks you execute before replanning (`candidates`, `generate`, `execute`), or detect and recover when a box slips.

## Benchmark to Beat

The full benchmark runs the 300 validation scenes with three policy seeds (`0`, `2`, `3`) and averages the results:

| Selection | ≥1 | ≥2 | ≥3 | 4 | Grasp | Mean boxes |
|---|---|---|---|---|---|---|
| one sample, no goal (`LatentPolicyAgent`) | 81.4% | 35.7% | 8.6% | 1.1% | 99.4% | 1.27 |
| 16 samples, subgoals every 80 steps, τ = 0.25 (`GoalPlanningAgent`) | 85.2% | 41.4% | 11.8% | 2.4% | 99.1% | 1.41 |

`≥1` to `4` are the percentages of episodes whose stack in the goal zone reaches at least that many boxes at some point (boxes standing side by side do not count as a stack), `Grasp` is the percentage of episodes in which the hand grasps a box, and `Mean boxes` is the average peak number of stacked boxes. Planning stacks 0.14 more boxes per episode on average (95% confidence interval 0.06–0.22).

Results vary noticeably with the policy seed (the no-planning baseline ranges from 1.21 to 1.31 mean boxes across the three seeds), so compare planners on the same scenes and seeds and report the average. The evaluation command is in Stage 5 of the [training track](../training/README.md).

Treat these 300 validation scenes as a development set. Competition details, including the held-out evaluation scenes, submission format, and compute limits, will be announced separately.
