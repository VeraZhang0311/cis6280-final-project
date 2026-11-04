#!/usr/bin/env bash
# Submit from the repository root; Liu GPUs first.
#SBATCH --job-name=cis6280-B1-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/b1-20-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${PHASE1_PYTHON:=/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_DISABLE_XET=1
export MPLCONFIGDIR="$PWD/outputs/.matplotlib"
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/outputs/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
"$PHASE1_PYTHON" -m unittest discover -s tests -p test_planning_diagnostics.py -v
"$PHASE1_PYTHON" -m assignment.training.planning.phase1 \
  --policy checkpoints/policy/df-smwm-c1-sigma04/policy.pt \
  --world-model checkpoints/jepa/smwm-c3/model.pt \
  --goals dataset/goals/scripted-val100-every80/goals.npz \
  --baselines B1 --scene-file notes/PHASE1/validation-scenes-20.json \
  --sample-seeds 0 2 3 --workers 4 --gpus 0 --videos 20 \
  --tail-state-steps 56 --keyframes 6 \
  --out "outputs/diagnostics/b1-20-${SLURM_JOB_ID}"
