#!/bin/bash
#SBATCH --job-name=6280-A5-collect
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
set -euo pipefail
TASK_ROOT=${TASK_ROOT:?Set TASK_ROOT to the isolated A5 workspace}
TASK_PYTHON=${TASK_PYTHON:-/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python}
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT:/mnt/projects/ll/tianzizh/cis6280-final-project/outputs/runtime-deps:/mnt/projects/ll/tianzizh/cis6280-final-project"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'SLURM_JOB_ID=%s PARTITION=%s NODE=%s RESTART_COUNT=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_PARTITION" "$SLURMD_NODENAME" "${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [[ ! -f dataset/reference-goals/goals.npz || ! -f dataset/reference-goals/goals.json ]]; then
    "$TASK_PYTHON" -u -m assignment.training.planning.goals \
        --scene-file inputs/collection-scenes.json --split train --every 80 \
        --horizon 1795 --workers 8 --gpus 0 --out dataset/reference-goals
fi
resume_args=()
if [[ -f dataset/onpolicy-a5/manifest.json ]]; then resume_args=(--resume); fi
exec "$TASK_PYTHON" -u -m assignment.training.data.collect_goal_onpolicy \
    --plan inputs/collection-plan.json --policy checkpoints/actor-120k.pt \
    --world-model checkpoints/original-wm.pt --goals dataset/reference-goals/goals.npz \
    --workers 8 --gpus 0 --encode-batch 16 --shard 16 --out dataset/onpolicy-a5 "${resume_args[@]}"
