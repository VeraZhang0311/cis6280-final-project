# Completed Planning Screen

Recorded on Penn PARCC, completed November 5, 2026. This snapshot contains all
eight configurations evaluated on the same 30 scenes and policy seeds 0, 2, 3:
90 episodes per configuration, 720 total. These are development-screen results;
the remaining 270-scene confirmation experiment has not been run.

## Contents

- `screen/<variant>/*.json`: 72 original reports, each containing ten episodes.
- `screen-summary.md` and `screen-summary.json`: regenerated aggregate results.
- `manifests/screen-task*.json`: all 24 completed array-task manifests, with scene
  lists, commands, configuration choices, and source-file SHA-256 hashes.
- `environment.txt`: dependency versions captured in the experimental environment.
- `artifacts.sha256`: hashes of the frozen world model, policy, and subgoal archive.
- `environment-tests.log`: the original successful 19-test run.

Reports and manifests retain original runtime paths and the original
`9540172...-dirty` code identifier because the experiment preceded this commit.
The manifests' source hashes identify the exact planner, evaluator, and runner
used; those source files are unchanged when packaging this snapshot. The editable
installation path in `environment.txt` is historical metadata, not a portable
installation command. See the setup script for installation on another machine.

The original GPU runs used array `10085491` and summary job `10085730`, all
COMPLETED with exit code 0:0. Per-task scheduling logs, videos, model weights,
datasets, partial runs, and generated source archives are not part of this snapshot.

## Recompute Without Models or a GPU

From the repository root, with Python 3.10 or newer:

```bash
python3 -m assignment.planning.summarize \
  --root results/planning/2026-11-05-screen --stage screen

python3 -m assignment.planning.report \
  --baseline results/planning/2026-11-05-screen/screen/baseline/*.json \
  --candidate results/planning/2026-11-05-screen/screen/trajectory/*.json
```

The summarizer requires complete scene/seed coverage and rejects duplicate
episodes and incompatible metadata. Its paired confidence intervals bootstrap
scenes after averaging policy-seed differences within each scene (10,000 draws,
seed 6280). Recorded artifact paths are not opened when recomputing statistics.

The observed best method is `trajectory`: mean peak stack 1.622 versus baseline
1.422, paired difference +0.200 with 95% CI [-0.022, 0.411]. The interval includes
zero; this screen does not establish a reliable improvement.

See [usage](../../../assignment/planning/README.md#group-planning-quick-start),
[protocol](../../../assignment/planning/EXPERIMENTS.md), and
[results interpretation](../../../assignment/planning/RESULTS.md).
