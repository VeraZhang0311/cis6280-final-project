#!/bin/bash
#SBATCH --job-name=6280-A123-round2-full
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
set -euo pipefail

TASK_ROOT=${TASK_ROOT:-/mnt/projects/ll/tianzizh/cis6280-v2-eval-r2-20261104}
TASK_PYTHON=${TASK_PYTHON:-/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python}
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT:/mnt/projects/ll/tianzizh/cis6280-final-project/outputs/runtime-deps:/mnt/projects/ll/tianzizh/cis6280-final-project"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

resume_args=()
if [[ -f outputs/planning/v2/full/run.json ]]; then
    resume_args=(--resume)
fi
printf 'SLURM_JOB_ID=%s PARTITION=%s NODE=%s RESTART_COUNT=%s\n' \
    "${SLURM_JOB_ID:-none}" "${SLURM_JOB_PARTITION:-none}" "${SLURMD_NODENAME:-none}" "${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec "$TASK_PYTHON" -u -m assignment.training.planning.evaluate_goal_actor \
    --policy checkpoints/goal-actor-r2-final.pt \
    --official-policy /mnt/projects/ll/tianzizh/cis6280-final-project/checkpoints/policy/df-smwm-c1-sigma04/policy.pt \
    --world-model /mnt/projects/ll/tianzizh/cis6280-final-project/checkpoints/jepa/smwm-c3/model.pt \
    --goals /mnt/projects/ll/tianzizh/cis6280-final-project/dataset/goals/scripted-val100-every80/goals.npz \
    --scene-file inputs/scenes-20.json --sample-seeds 0 2 3 --groups A1 A2 A3 \
    --horizon 1795 --workers 8 --gpus 0 --videos 20 --out outputs/planning/v2/full \
    "${resume_args[@]}"
