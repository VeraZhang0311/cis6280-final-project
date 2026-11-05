#!/usr/bin/env bash
# Submit from the repository root with a verified GPU partition, for example:
# sbatch --parsable --partition=dgx-b200 --qos=dgx --gres=gpu:1 --cpus-per-task=28 \
#   --mem=224G --time=04:00:00 --output=runs/planning/%x-%j.log \
#   scripts/planning.slurm smoke baseline combined
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
stage="${1:-smoke}"
if (( $# )); then shift; fi
if (( $# == 0 )); then set -- baseline combined; fi
nvidia-smi
array_args=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    array_args=(--task-index "$SLURM_ARRAY_TASK_ID")
fi
.venv/bin/python -m assignment.planning.experiments --stage "$stage" \
    --variants "$@" --workers "${PLANNING_WORKERS:-4}" \
    --shard-size "${PLANNING_SHARD_SIZE:-10}" "${array_args[@]}" --run
