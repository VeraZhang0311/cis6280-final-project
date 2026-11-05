#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PLANNING_MODEL_ROOT="${PLANNING_MODEL_ROOT:-/vast/projects/xuhw/healthworldx/zhangrm/model/cis6280-stack-world-models-v2}"
export PLANNING_DATA_ROOT="${PLANNING_DATA_ROOT:-/vast/projects/xuhw/healthworldx/zhangrm/data/cis6280-stack-datasets-v2}"
if [[ ! -x .venv/bin/python ]]; then python3.12 -m venv --without-pip .venv; fi
python3.12 -m pip --python .venv/bin/python install --no-compile --progress-bar off pip -e . 'torch==2.9.1' 'mujoco==3.13.0'
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
import os
model_root = os.environ['PLANNING_MODEL_ROOT']
data_root = os.environ['PLANNING_DATA_ROOT']
revisions = {'model': '4719eb832d8aea448aae9e4eb79f219ca73452f4',
             'dataset': '3709e87e9e00838b96e7854c8db463025fd38b7a'}
for repo, kind, filename, directory in [
    ('YongYong/cis6280-stack-world-models-v2', 'model', 'jepa/smwm-c3/model.pt', model_root),
    ('YongYong/cis6280-stack-world-models-v2', 'model', 'policy/df-smwm-c1-sigma04/policy.pt', model_root),
    ('YongYong/cis6280-stack-datasets-v2', 'dataset', 'goals/scripted-val300-every80/goals.npz', data_root),
]:
    hf_hub_download(repo, filename, repo_type=kind, revision=revisions[kind], local_dir=directory)
PY
mkdir -p runs/planning
.venv/bin/python -m pip freeze > runs/planning/environment.txt
sha256sum "$PLANNING_MODEL_ROOT/jepa/smwm-c3/model.pt" "$PLANNING_MODEL_ROOT/policy/df-smwm-c1-sigma04/policy.pt" "$PLANNING_DATA_ROOT/goals/scripted-val300-every80/goals.npz" > runs/planning/artifacts.sha256
