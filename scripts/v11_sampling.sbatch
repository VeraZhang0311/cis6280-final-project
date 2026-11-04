#!/usr/bin/env bash
#SBATCH --job-name=cis6280-G6G7
#SBATCH --gres=gpu:geforce_rtx_2080_ti:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=outputs/v11-sampling-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
py=/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/outputs/.matplotlib"
export PYTHONPATH="$PWD:$PWD/outputs/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
"$py" -m unittest discover -s tests -p 'test_v11*.py' -v
"$py" -m unittest discover -s tests -p test_planning_diagnostics.py -v
out="outputs/planning/v1.1/sampling-${SLURM_JOB_ID}"
resume=()
if [[ -f "$out/run.json" ]]; then resume=(--resume); fi
"$py" -m assignment.training.planning.v11_sampling \
  --reference-run /mnt/projects/ll/tianzizh/cis6280-v11-bidir-r2/outputs/planning/v1.1/bidir-1498961 \
  --out "$out" --workers 4 --gpus 0,1 --fresh-reference "${resume[@]}"
