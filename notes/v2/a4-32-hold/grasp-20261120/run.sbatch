#!/bin/bash
#SBATCH --job-name=6280-emoo-hold-180
#SBATCH --account=liu-account
#SBATCH --partition=liu-compute
#SBATCH --qos=ll-med
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
set -euo pipefail
: "${TASK_ROOT:?Set the isolated workspace}"
cd "$TASK_ROOT"
bash scripts/evaluate_a4_32_hold.sbatch
export PYTHONPATH="$TASK_ROOT:/mnt/projects/ll/tianzizh/cis6280-final-project/outputs/runtime-deps"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python audit_hold.py \
    outputs/planning/v2/a4-32-local-hold \
    --historical /mnt/projects/ll/tianzizh/cis6280-v2-integration-20261110-01/outputs/planning/v2/integration
