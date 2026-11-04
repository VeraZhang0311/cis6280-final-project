#!/usr/bin/env bash
#SBATCH --job-name=cis6280-postfirst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=outputs/openloop-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${PHASE1_PYTHON:=/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLCONFIGDIR="$PWD/outputs/.matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/outputs/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
"$PHASE1_PYTHON" -m unittest discover -s tests -p 'test_openloop*.py' -v
exec "$PHASE1_PYTHON" -m assignment.training.planning.openloop \
  --source-run outputs/diagnostics/b1-20-1497072 --workers 4 --gpus "${OPENLOOP_GPUS:-0}" \
  --out "outputs/openloop/postfirst-${SLURM_JOB_ID}"
