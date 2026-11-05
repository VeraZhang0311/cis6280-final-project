#!/bin/bash
# Submit with --dependency=afterok:<collection-job>.
#SBATCH --job-name=6280-A5-WM-60k
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
set -euo pipefail
TASK_ROOT=${TASK_ROOT:?Set TASK_ROOT to the isolated A5 workspace}
TASK_PYTHON=${TASK_PYTHON:-/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python}
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT:/mnt/projects/ll/tianzizh/cis6280-v2-actor-20261104/runtime-deps:/mnt/projects/ll/tianzizh/cis6280-final-project/outputs/runtime-deps:/mnt/projects/ll/tianzizh/cis6280-final-project"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
printf 'SLURM_JOB_ID=%s PARTITION=%s NODE=%s RESTART_COUNT=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_PARTITION" "$SLURMD_NODENAME" "${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv
resume_args=()
if [[ -f checkpoints/wm-a5/train-state.pt ]]; then resume_args=(--resume); fi
exec "$TASK_PYTHON" -u -m assignment.training.training.finetune_goal_onpolicy \
    --data dataset/onpolicy-a5 --init checkpoints/original-wm.pt --policy checkpoints/actor-120k.pt \
    --out checkpoints/wm-a5 --steps 60000 --batch 256 --microbatch 256 \
    --lr 1e-4 --warmup 1000 --seed 0 --eval-every 2000 --state-every 200 \
    --wandb-project 6280_wm --wandb-entity tianzizh-university-of-pennsylvania \
    --wandb-run-name a5-wm-onpolicy-120k --wandb-run-id a5-wm-onpolicy-120k-s0 \
    "${resume_args[@]}"
