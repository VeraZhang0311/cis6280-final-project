#!/bin/bash
# Explicitly submit only after the WM selection receipt and both A4 groups are complete.
#SBATCH --job-name=6280-A5-16-32-full
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
set -euo pipefail
TASK_ROOT=${TASK_ROOT:?Set TASK_ROOT to the isolated A5 workspace}
TASK_PYTHON=${TASK_PYTHON:-/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python}
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT:/mnt/projects/ll/tianzizh/cis6280-final-project/outputs/runtime-deps:/mnt/projects/ll/tianzizh/cis6280-final-project"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
resume_args=()
if [[ -f outputs/planning/a5/run.json ]]; then resume_args=(--resume); fi
exec "$TASK_PYTHON" -u -m assignment.training.planning.evaluate_goal_actor \
    --policy checkpoints/actor-120k.pt --world-model checkpoints/wm-a5/model.pt \
    --wm-selection checkpoints/wm-a5/selection.json \
    --goals /mnt/projects/ll/tianzizh/cis6280-final-project/dataset/goals/scripted-val100-every80/goals.npz \
    --scene-file inputs/development-scenes.json --sample-seeds 0 2 3 --groups A5-16 A5-32 \
    --reference-run /mnt/projects/ll/tianzizh/cis6280-v2-a4-eval120k-20261104/outputs/planning/v2/a4-120k \
    --horizon 1795 --workers 8 --gpus 0 --videos 20 --out outputs/planning/a5 "${resume_args[@]}"
