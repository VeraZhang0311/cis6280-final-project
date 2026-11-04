#!/usr/bin/env bash
# Submit from the project root, with partition/account/QOS chosen at submission time.
# sbatch -p liu-compute -A liu-account --qos=ll-med scripts/phase1_cluster.sbatch
# If Liu has no available GPU: sbatch -p batch -A liu-account --qos=normal ...
#SBATCH --job-name=cis6280-phase1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/phase1-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
: "${PHASE1_PYTHON:=$PWD/.venv/bin/python}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_DISABLE_XET=1
export MPLCONFIGDIR="$PWD/outputs/.matplotlib"
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/outputs/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs
"$PHASE1_PYTHON" -m unittest discover -s tests -p test_planning_diagnostics.py -v
exec "$PHASE1_PYTHON" -m assignment.training.planning.phase1 \
    --policy checkpoints/policy/df-smwm-c1-sigma04/policy.pt \
    --world-model checkpoints/jepa/smwm-c3/model.pt \
    --goals dataset/goals/scripted-val100-every80/goals.npz \
    --baselines B1 --scene-seeds 100038 --sample-seeds 0 2 3 --workers 2 --gpus 0 --videos 1 \
    --out "outputs/diagnostics/scene100038-${SLURM_JOB_ID}"
