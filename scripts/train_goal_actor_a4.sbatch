#!/bin/bash
#SBATCH --job-name=6280-A4-tail-actor
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
set -euo pipefail

TASK_ROOT=${TASK_ROOT:-/mnt/projects/ll/tianzizh/cis6280-v2-a4-20261104}
TASK_PYTHON=${TASK_PYTHON:-/mnt/projects/ll/tianzizh/hoi_data_pipeline_runtime_20261029/bin/python}
RUN_NAME=goal-actor-v2-a4-tail-seed0-20261104
PARENT=/mnt/projects/ll/tianzizh/cis6280-v2-eval-r2-20261104/checkpoints/goal-actor-r2-final.pt
DERIVED=/mnt/projects/ll/tianzizh/cis6280-v2-actor-20261104/dataset/derived
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT:/mnt/projects/ll/tianzizh/cis6280-v2-actor-20261104/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_CONSOLE=off
export WANDB_DISABLE_CODE=true

resume_args=()
if [[ -f "checkpoints/$RUN_NAME/train-state.pt" ]]; then
    resume_args=(--resume)
fi
printf 'SLURM_JOB_ID=%s PARTITION=%s NODE=%s RESTART_COUNT=%s\n' \
    "${SLURM_JOB_ID:-none}" "${SLURM_JOB_PARTITION:-none}" "${SLURMD_NODENAME:-none}" "${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec "$TASK_PYTHON" -u -m assignment.training.training.train_goal_policy \
    --cache "$DERIVED/jepa-cache-4cube-20k" --latents "$DERIVED/latents-smwm-c3" \
    --init-goal-policy "$PARENT" --include-tail \
    --expected-init-sha256 d039fbba25eabb9b976fff945afd6b59031cbbf11df5f916a5671dd774e234bf \
    --expected-prior-steps 100000 \
    --encoder /mnt/projects/ll/tianzizh/cis6280-final-project/checkpoints/jepa/smwm-c3/model.pt \
    --out "checkpoints/$RUN_NAME" --log-dir "outputs/training/v2/$RUN_NAME" \
    --steps 50000 --batch 256 --microbatch 64 --lr 1e-4 --warmup 1000 \
    --ema 0.999 --seed 0 --eval-every 1000 --save-every 5000 --state-every 1000 \
    --wandb-project 6280_actor --wandb-entity tianzizh-university-of-pennsylvania \
    --wandb-run-name "$RUN_NAME" --wandb-run-id v2actor-20261104-a4-tail-s0 \
    "${resume_args[@]}"
