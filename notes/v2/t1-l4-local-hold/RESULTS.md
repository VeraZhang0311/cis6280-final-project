# T1-L4 HOLD optimization results

Updated November 23, 2026 (America/New_York).

The corrected Colab screen is complete for scene `100183`, sample seed `3`.
The matched L40/BF16 evaluation over 20 scenes and sample seeds `0/2/3` has not
run. This screen is mechanism evidence only and does not establish a general
improvement.

## Corrected controlled screen

Evidence directory:
`/content/drive/MyDrive/cis6280-final-project-v2/outputs/t1-l4-local-hold/20261123-233249`.

All groups use exactly the released T1-L4 selector before two consecutive real
final-goal confirmations. The latch is permanent. Only the post-latch selector
differs.

| Method | Post-latch selector | Peak | Final | Success | Four-box retention | Mean decision ms | Decisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A4-32-T1-L4-HOLD` | Mean distance over blocks 13--16 | 4 | 4 | 1 | 1/1 | 186.6 | 282 |
| `A4-32-T1-L4-FIRST4-HOLD` | Mean distance over blocks 1--4 | 4 | 4 | 1 | 1/1 | 187.0 | 284 |
| `A4-32-T1-L4-LOCAL-HOLD` | Four-block safety and wrist RMS, with first-four mean fallback | 4 | 4 | 1 | 1/1 | 185.8 | 284 |

All paired differences in peak, final count, and success are zero. The original
late-four HOLD terminated two planning decisions earlier in this screen. The
small latency differences are not evidence of a runtime difference.

## HOLD audit

| Method | Latched decisions | Entries | Exits | Decisions with a safe candidate | Selected wrist RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A4-32-T1-L4-HOLD` | 5 | 1 | 0 | Not applicable | Not applicable |
| `A4-32-T1-L4-FIRST4-HOLD` | 7 | 1 | 0 | Not applicable | Not applicable |
| `A4-32-T1-L4-LOCAL-HOLD` | 7 | 1 | 0 | 0/7 (0%) | 0.0699 |

No LOCAL-HOLD decision had a candidate whose first four imagined blocks were
all strictly within the final-goal threshold. Every LOCAL-HOLD decision used
the `first4_mean` fallback. LOCAL-HOLD was therefore behaviorally equivalent to
FIRST4-HOLD in this screen. The reported wrist RMS is diagnostic only; wrist
RMS did not determine any selected candidate.

## Interpretation

- Pre-latch equivalence is supported: all groups reached each stacking stage at
  the same reported transition times.
- FIRST4-HOLD did not improve peak, final count, success, or retention over the
  released late-four HOLD in this screen.
- Strict local safety and wrist-RMS ranking were not exercised, so this screen
  provides no evidence for those mechanisms.
- The corrected LOCAL-HOLD optimization is not yet validated. It was not worse
  on the official outcome metrics in this case, but it did not improve them.

The historical integration run used an NVIDIA L40 with bfloat16. This Colab
screen used a T4-class runtime, which follows a different numerical precision
path. The historical `100183 / 3` episode had 64 latched decisions, peak four,
final two, and no success; the Colab screen had only 5--7 latched decisions and
all groups succeeded. The within-screen comparison remains controlled, but the
Colab trajectory is not a strict replay of the historical L40/BF16 episode.

## Frozen follow-up

The next valid decision point is the frozen 20-scene x 3-seed matched evaluation
on L40/BF16. It compares the released HOLD, FIRST4-HOLD, and LOCAL-HOLD over 60
episodes per group. Primary metrics are success, mean final boxes, and
`P(final=4 | peak=4)`. The mechanism audit must also report the safe-candidate
rate and fallback rate. No selector parameters should be changed after this
screen.

## Superseded A4-based screen

The earlier scene `100038`, seed `0` screen used A4-32 `active_min` before the
latch rather than the released T1-L4 objective. It is retained as structured
debugging evidence in `evidence/single-scene-summary.json`, explicitly marked
`superseded_debugging_screen`; it is not evidence for this optimization.
