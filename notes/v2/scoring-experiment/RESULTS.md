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

## Closed-loop screening result — 2026-11-06

The Colab run completed all three predeclared groups on validation scene
`100038`, sample seed `0`, for the full 1795-control-step horizon. Each group
made 359 planning decisions. The run took 1013.22 seconds (16 minutes 53
seconds) in total.

| Group | Peak boxes | Final boxes | Official success | Mean decision | P95 decision |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 `A4-32` | 4 | 1 | 0 | 185.02 ms | 248.53 ms |
| `A4-32-T1` | 4 | **3** | 0 | 186.35 ms | 245.11 ms |
| `A4-32-T2` | 3 | 1 | 0 | 188.79 ms | 244.94 ms |

All groups used exactly the same computational structure: 359 encoder calls,
359 policy samples, 3,590 policy-velocity calls, 359 action-encoder calls, 359
WM rollouts, and 5,744 predictor calls. This confirms that the comparison
changes candidate scoring rather than candidate count, rollout depth, or model
usage.

### Paired differences from the v2 baseline

For T1 minus v2, peak boxes changed by `0`, final boxes changed by `+2`, and
official success changed by `0`. Expressed descriptively for this episode, T1
raised final retention from 1/4 to 3/4 boxes (25% to 75%) with a 1.33 ms, or
approximately 0.72%, increase in mean decision time. It did **not** increase the
maximum number of boxes reached and did **not** convert the episode into an
official success.

For T2 minus v2, peak boxes changed by `-1`, final boxes by `0`, and success by
`0`. Relative to T1, T2 lost one peak box and two final boxes. Increasing the
terminal coefficient from 0.25 to 0.50 therefore hurt this screening episode;
there is no evidence here that stronger terminal weighting improves retention.

### Progress and retention behavior

The v2 baseline first-to-second and third-to-fourth transitions were observed
at 35.0 and 36.2 seconds. T1 reached the same transitions at 42.8 and 33.1
seconds: it was slower early but slightly faster on the final transition. T2
took 65.1 seconds for the first-to-second transition and never converted the
third-to-fourth transition.

Both v2 and T1 reached four boxes transiently, yet neither finished with four.
The main observed gain is therefore retention after progress, not a higher
attainable peak. The remaining failure mode is also clear: after reaching the
four-box state, continued closed-loop actions can destroy a completed or nearly
completed arrangement. A success latch, verified completion stop, or explicit
post-completion hold policy remains a separate future hypothesis and was not
added to this controlled scoring experiment.

The report's `lookahead.applicable_decisions=0` is expected. That field belongs
to the separate next-goal lookahead mechanism. All three groups still perform
the 16-block active-goal WM rollout, as confirmed by the identical rollout and
predictor call counts.

## Retention decision

`A4-32-T1` is retained as the current scoring version:

- time weight: `0.05`;
- terminal weight: `0.25`;
- normalized by the active reference gap;
- 32 candidates and a 16-block WM rollout;
- execute only the first block, then reobserve and replan;
- stable earliest-candidate tie breaking.

The v2 `A4-32` and T2 code paths are preserved as reproducible controls. T2 is
not promoted. This decision is based on final-box retention in the locked
screening episode and the absence of a meaningful latency penalty, not on an
official-success improvement.

## Evidence and limitations

The exact reported summary is preserved in
[evidence/single-scene-summary.json](evidence/single-scene-summary.json). It is a
structured transcription of the completed Colab summary supplied for this run;
the generated Drive `run.json`, per-step traces, and videos remain the
authoritative runtime artifacts.

This is one scene, one seed, and one episode per group. Consequently all paired
95% confidence intervals are `null`: there is no scene-level variance estimate.
The result supports the narrow statement that T1 outperformed v2 and T2 on
final retention in scene `100038`, seed `0`. It does not establish statistical
significance or generalization. A confirmation run must use a frozen multi-scene
list and must not retune the weights from those results.

