"""Build a self-contained Colab experiment by refactoring the official notebook.

All new Python source is embedded, so no GitHub push or private-repo token is
needed. Re-run this builder whenever the agent, evaluator or tests change.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def install_source():
    paths = ['assignment/training/planning/agent.py', 'assignment/planning/horizon_experiment.py',
             'tests/test_planning_horizon.py']
    sources = {path: (ROOT / path).read_text() for path in paths}
    return '''# Install the exact experiment source embedded in this notebook.
# This writes code only; the downloaded model and goals files are never edited.
import importlib, json, subprocess, sys
from pathlib import Path
sources = json.loads(''' + repr(json.dumps(sources)) + ''')
for relative, source in sources.items():
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
import assignment.training.planning.agent as agent_module
importlib.reload(agent_module)
subprocess.check_call([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                       '-p', 'test_planning_horizon.py', '-v'], cwd=ROOT)
from assignment.planning.horizon_experiment import run_experiment, summarize
print('Experiment source installed; horizon tests passed.')
'''


def main():
    notebook = json.loads((ROOT / 'assignment/planning/colab.ipynb').read_text())
    original = copy.deepcopy(notebook['cells'])
    notebook['cells'] = [cell('markdown', '''# Shafaa: 2-, 4-, and 8-second planning horizons

Branch: `shafaa`, based on Tianzi v2 commit `3638812`.
Use **Runtime → Change runtime type → T4 GPU**, then **Run all**.

This notebook runs three full attempts on scene 100038, seed 0, and displays their videos.
Allow roughly 15–25 minutes on a T4. This is a demonstration, not a benchmark conclusion.

Only the world-model prediction horizon changes (4/8/16 blocks). Every variant samples
16 candidates of 16 blocks, uses terminal squared latent distance and the original
goal-switching/action rules, and executes one block (5 control steps / 0.5 seconds).
The original policy, world model and official 8-second subgoal images stay unchanged.
No goal-conditioned v2 trained actor, training, optimizer, or privileged planner input is used.
''')]
    notebook['cells'] += [original[1], original[2]]
    notebook['cells'].append(cell('code', '''# Download original inputs; pass the WM path explicitly instead of editing policy.pt.
import shutil
from huggingface_hub import hf_hub_download
CKPT = Path('/content/horizon-inputs')
CKPT.mkdir(parents=True, exist_ok=True)
WM, POLICY, GOALS = CKPT/'model.pt', CKPT/'policy.pt', CKPT/'goals.npz'
for repo, name, dest, kind in [
    ('YongYong/cis6280-stack-world-models-v2', 'jepa/smwm-c3/model.pt', WM, 'model'),
    ('YongYong/cis6280-stack-world-models-v2', 'policy/df-smwm-c1-sigma04/policy.pt', POLICY, 'model'),
    ('YongYong/cis6280-stack-datasets-v2', 'goals/scripted-val100-every80/goals.npz', GOALS, 'dataset'),
]:
    shutil.copyfile(hf_hub_download(repo, name, repo_type=kind), dest)
print('Original frozen inputs ready.')
'''))
    notebook['cells'].append(cell('code', install_source()))
    notebook['cells'].append(cell('code', '''# Real-model regression: explicit 8-second prediction must match the official class.
import numpy as np
reference_scope = {}
exec(compile(''' + repr(''.join(original[7]['source'])) + ''', '<official-notebook-planner>', 'exec'), reference_scope)
OfficialGoalPlanningAgent = reference_scope['GoalPlanningAgent']
common = dict(world_model=str(WM), candidates=16, generate=16, execute=1, n_steps=10, lag=0, tau=.25)
official = OfficialGoalPlanningAgent(str(POLICY), 'cuda', **common)
ours = agent_module.GoalPlanningAgent(str(POLICY), 'cuda', planning_horizon=16, **common)
with np.load(GOALS) as d:
    i = list(d['seeds']).index(100038)
    reference_goals = d['sub_frames'][i, :int(d['sub_count'][i])].copy()
    reference_init = d['init_frames'][i].copy()
for a in (official, ours):
    a.reset(reference_goals, reference_init, seed=100038)
for observation in [reference_init, *reference_goals[:3]]:
    np.testing.assert_array_equal(official.step(observation), ours.step(observation))
    np.testing.assert_array_equal(official.last_distance.cpu().numpy(), ours.last_distance.cpu().numpy())
del official, ours
torch.cuda.empty_cache()
print('PASS: 8-second actions and candidate costs exactly match the original planner on 4 observations.')
'''))
    notebook['cells'].append(cell('markdown', '''## Configure the comparison

Default: one scene × one seed × three horizons = three full episodes.
Use `EPISODE_STEPS = 40` only for a quick implementation check, with a fresh output folder.
For a later development comparison, predeclare matching validation scenes and use seeds [0, 2, 3].
Keep `PLANNING_SECONDS = [2, 4, 8]`, policy generation and official goal images fixed.
'''))
    notebook['cells'].append(cell('code', '''from datetime import datetime, timezone
SCENES = [100038]
SAMPLE_SEEDS = [0]
PLANNING_SECONDS = [2, 4, 8]
EPISODE_STEPS = 1795
OUT = Path('/content/horizon-results') / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
print('Results folder:', OUT)
'''))
    notebook['cells'].append(cell('code', '''rows = run_experiment(
    POLICY, WM, GOALS, OUT / 'full', scenes=SCENES, sample_seeds=SAMPLE_SEEDS,
    planning_seconds=PLANNING_SECONDS, episode_steps=EPISODE_STEPS,
)
'''))
    notebook['cells'].insert(-1, cell('code', '''# A brief end-to-end check of all horizons before the full experiment.
smoke_rows = run_experiment(POLICY, WM, GOALS, OUT / 'smoke', scenes=[100038],
                            sample_seeds=[0], episode_steps=40)
print('All 3 short smoke runs passed; these are not performance results.')
'''))
    notebook['cells'].append(cell('code', '''from IPython.display import Video, display, Markdown
groups = summarize(rows)
table = '| Horizon | Episodes | Mean peak | Mean final | Success | Mean decision ms |\\n'
table += '|---|---:|---:|---:|---:|---:|\\n'
for r in groups:
    table += f"| {r['planning_seconds']}s | {r['episodes']} | {r['mean_peak_boxes']:.2f} | {r['mean_final_boxes']:.2f} | {r['success_rate']:.1%} | {r['mean_decision_ms']:.1f} |\\n"
display(Markdown(table))
if EPISODE_STEPS != 1795:
    display(Markdown('**Short smoke test only: do not interpret box counts as performance.**'))
for row in rows:
    display(Markdown(f"**{row['planning_seconds']}s horizon — scene {row['scene']}, seed {row['sample_seed']}**"))
    display(Video(row['video'], embed=True, width=512))
'''))
    notebook['cells'].append(cell('markdown', '''## Save your evidence

Run the cell below to download a ZIP containing videos, per-decision traces, model/goal
SHA256 hashes, source hashes, settings, episode results and the summary. Colab files
are temporary until you download them. A rerun with identical settings and `OUT` resumes
completed episodes; changing settings/source requires a fresh output folder.

Questions for later analysis: does a shorter horizon improve box counts, reduce decision
time, or become too shortsighted to reach a distant subgoal? One scene cannot establish
an improvement. Compare matching scenes/seeds and keep development and confirmation separate.
'''))
    notebook['cells'].append(cell('code', '''import shutil
from google.colab import files
archive = shutil.make_archive(str(OUT), 'zip', root_dir=OUT)
files.download(archive)
'''))
    for i, c in enumerate(notebook['cells']):
        c['id'] = f'horizon-{i:02d}'
        if c['cell_type'] == 'code':
            c['execution_count'], c['outputs'] = None, []
    output = ROOT / 'assignment/planning/horizon_colab.ipynb'
    output.write_text(json.dumps(notebook, indent=1) + '\n')
    print(output)


if __name__ == '__main__':
    main()
