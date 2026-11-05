#!/bin/bash
#SBATCH --job-name=6280-A4-16-32-full
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
set -euo pipefail

TASK_ROOT=${TASK_ROOT:-/mnt/projects/ll/tianzizh/cis6280-v2-a4-20261104}
TASK_PYTHON=${TASK_PYTHON:-/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python}
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT:/mnt/projects/ll/tianzizh/cis6280-final-project/outputs/runtime-deps:/mnt/projects/ll/tianzizh/cis6280-final-project"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

resume_args=()
if [[ -f outputs/planning/v2/a4-full/run.json ]]; then
    resume_args=(--resume)
fi
exec "$TASK_PYTHON" -u -m assignment.training.planning.evaluate_goal_actor \
    --policy checkpoints/goal-actor-v2-a4-tail-seed0-20261104/policy.pt \
    --world-model /mnt/projects/ll/tianzizh/cis6280-final-project/checkpoints/jepa/smwm-c3/model.pt \
    --goals /mnt/projects/ll/tianzizh/cis6280-final-project/dataset/goals/scripted-val100-every80/goals.npz \
    --official-policy /mnt/projects/ll/tianzizh/cis6280-final-project/checkpoints/policy/df-smwm-c1-sigma04/policy.pt \
    --scene-file inputs/scenes-20.json --sample-seeds 0 2 3 --groups A4-16 A4-32 \
    --reference-run /mnt/projects/ll/tianzizh/cis6280-v2-eval-r2-20261104/outputs/planning/v2/full \
    --reference-launch inputs/A2-launch.json \
    --horizon 1795 --workers 8 --gpus 0 --videos 20 --out outputs/planning/v2/a4-full \
    "${resume_args[@]}"
