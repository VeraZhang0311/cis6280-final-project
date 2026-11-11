# A4-32-HOLD experiment

This experiment isolates the final-goal HOLD mechanism from the weaker T1/L4
approach score. Before the latch, `A4-32-HOLD` is exactly the released A4-32
selector: 32 candidates ranked by minimum active-goal distance over all 16
imagined blocks. It uses the same actor, world model, candidate budget, rollout,
goal switching, and one-block execution.

The latch activates after two consecutive real observations are within the
strict threshold for the final official goal. It remains active for the rest of
the episode. After latching, candidates are ranked by their mean distance over
imagined blocks 13--16, favoring sustained final-goal retention. Simulator box
counts, reward, and success are never read by the planner.

Run a fresh `A4-32` control and `A4-32-HOLD` on the frozen 20 development scenes
and sample seeds 0/2/3. Primary metrics are official success, mean final boxes,
and four-box retention. Secondary metrics are mean peak boxes, third-to-fourth
conversion, HOLD trigger count, false-positive HOLD outcomes, and decision time.
This is a paired development experiment, not independent confirmation.
