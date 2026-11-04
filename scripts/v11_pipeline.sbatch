#!/usr/bin/env bash
# New protocol: bidirectional G4 and combined G5; old G4 is excluded.
#SBATCH --job-name=cis6280-v11-bidir
#SBATCH --gres=gpu:geforce_rtx_2080_ti:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/v11-bidir-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${V11_PYTHON:=/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python}"
: "${V11_WORKERS:=8}"
: "${V11_GPUS:=0,1,2,3}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/outputs/.matplotlib"
export PYTHONPATH="$PWD:$PWD/outputs/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
"$V11_PYTHON" -m unittest discover -s tests -p 'test_v11*.py' -v
"$V11_PYTHON" -m unittest discover -s tests -p test_planning_diagnostics.py -v
out="outputs/planning/v1.1/bidir-${SLURM_JOB_ID}"
resume=()
if [[ -f "$out/run.json" ]]; then resume=(--resume); fi
"$V11_PYTHON" -m assignment.training.planning.v11 \
  --source-run /mnt/projects/ll/tianzizh/cis6280-final-project/outputs/diagnostics/b1-20-1497072 \
  --reuse-run /mnt/projects/ll/tianzizh/cis6280-v11-20261103/outputs/planning/v1.1/run-1498776 \
  --out "$out" --workers "$V11_WORKERS" --gpus "$V11_GPUS" "${resume[@]}"
