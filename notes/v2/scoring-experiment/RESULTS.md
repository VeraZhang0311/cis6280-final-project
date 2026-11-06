# A4-32 trajectory-scoring results

## Implementation verification — 2026-11-06

Command:

```text
OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.11 -m unittest discover -s tests -p test_goal_actor.py -v
```

Environment: Python 3.11, NumPy 2.4.6, PyTorch 2.14.0, CPU.

Primary Goal Actor result: **17/17 tests passed** in 1.051 seconds.

Full v2 regression verification:

```text
test_goal_*.py:     29/29 passed
test_onpolicy_a5.py: 10/10 passed
total:              39/39 passed
```

The final snapshot/restore test used MuJoCo 3.17.0 and ImageIO 2.38.1 from an
isolated `/tmp` dependency directory; no repository or system dependency files
were changed. All six code cells in the persistent notebook also passed Python
AST parsing, and `git diff --check` reported no whitespace errors.

The tests verify:

- exact T1/T2 formulas and reference-gap normalization;
- predeclared `0.05`, `0.25`, and `0.50` weights;
- stable earliest-candidate tie breaking;
- 32-candidate budget and one 16-step WM rollout per decision;
- action clipping/history alignment and real-observation-only goal switching;
- JSON-safe traces containing arrival blocks, terminal distances, and costs;
- paired comparisons for T1−R0, T2−R0, and T2−T1;
- all existing A1–A5 behavior and resume/reference checks remain passing.

## Closed-loop result

Status: **pending Colab GPU execution**. No task-performance claim is made from
the unit tests. Paste the generated output path and copy `run.json` and
`summary.json` into a new evidence subdirectory after the run completes; do not
edit those generated files.

