#!/bin/bash
#SBATCH --job-name=6280-integration-smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
: "${TASK_ROOT:?Set TASK_ROOT to the isolated integration workspace}"
TASK_PYTHON=${TASK_PYTHON:-/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python}
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT:/mnt/projects/ll/tianzizh/cis6280-final-project/outputs/runtime-deps"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mapfile -t groups < <("$TASK_PYTHON" -c 'import json; print("\n".join(json.load(open("notes/v2/integration/plan.json"))["groups"]))')
exec "$TASK_PYTHON" -u -m assignment.training.planning.evaluate_goal_actor \
    --policy /mnt/projects/ll/tianzizh/cis6280-v2-a4-eval120k-20261104/checkpoints/goal-actor-a4-120k-ema.pt \
    --a4-cumulative-step 120000 \
    --world-model /mnt/projects/ll/tianzizh/cis6280-final-project/checkpoints/jepa/smwm-c3/model.pt \
    --goals /mnt/projects/ll/tianzizh/cis6280-final-project/dataset/goals/scripted-val100-every80/goals.npz \
    --scenes 100038 --sample-seeds 0 --groups "${groups[@]}" \
    --horizon 20 --workers 1 --gpus 0 --videos 1 --out outputs/planning/v2/integration-smoke
