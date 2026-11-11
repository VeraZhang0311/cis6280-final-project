#!/bin/bash
#SBATCH --job-name=6280-a4-local-hold
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
set -euo pipefail

: "${TASK_ROOT:?Set TASK_ROOT to the isolated A4-32-HOLD workspace}"
TASK_PYTHON=${TASK_PYTHON:-/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python}
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT:/mnt/projects/ll/tianzizh/cis6280-final-project/outputs/runtime-deps"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

resume_args=()
if [[ -f outputs/planning/v2/a4-32-local-hold/run.json ]]; then
    resume_args=(--resume)
fi
mapfile -t groups < <("$TASK_PYTHON" -c 'import json; print("\n".join(json.load(open("notes/v2/a4-32-hold/plan.json"))["groups"]))')
nvidia-smi --query-gpu=name,memory.total --format=csv
"$TASK_PYTHON" -u -m assignment.training.planning.evaluate_goal_actor \
    --policy /mnt/projects/ll/tianzizh/cis6280-v2-a4-eval120k-20261104/checkpoints/goal-actor-a4-120k-ema.pt \
    --a4-cumulative-step 120000 \
    --world-model /mnt/projects/ll/tianzizh/cis6280-final-project/checkpoints/jepa/smwm-c3/model.pt \
    --goals /mnt/projects/ll/tianzizh/cis6280-final-project/dataset/goals/scripted-val100-every80/goals.npz \
    --official-policy /mnt/projects/ll/tianzizh/cis6280-final-project/checkpoints/policy/df-smwm-c1-sigma04/policy.pt \
    --scene-file notes/v2/a4-32-hold/plan.json --sample-seeds 0 2 3 --groups "${groups[@]}" \
    --experiment-plan notes/v2/a4-32-hold/plan.json \
    --horizon 1795 --workers 8 --gpus 0 --videos 20 --out outputs/planning/v2/a4-32-local-hold \
    "${resume_args[@]}"
exec "$TASK_PYTHON" scripts/report_integration.py outputs/planning/v2/a4-32-local-hold
