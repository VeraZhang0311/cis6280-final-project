"""Metrics for a single fixed imagined trajectory and its actual open-loop execution."""
import numpy as np


def candidate_goal_distances(imagined, goal):
    """Return (candidates, future blocks) squared distances, excluding the initial context."""
    imagined = np.asarray(imagined, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if imagined.ndim != 3 or goal.shape != (imagined.shape[-1],):
        raise ValueError('expected imagined (N,H,D) and one fixed goal (D,)')
    if not imagined.shape[0] or not imagined.shape[1] or not np.isfinite(imagined).all() or not np.isfinite(goal).all():
        raise ValueError('candidate trajectories and goal must be finite and nonempty')
    return np.square(imagined-goal).sum(-1)


def aligned_trajectory_metrics(imagined, actual, goal):
    """actual[0] must be the observation AFTER executing the first complete block."""
    imagined, actual, goal = (np.asarray(x, dtype=np.float64) for x in (imagined, actual, goal))
    if imagined.ndim != 2 or imagined.shape != actual.shape or goal.shape != (imagined.shape[-1],):
        raise ValueError('expected matching future-only (H,D) latents and fixed goal (D,)')
    if not len(imagined) or not all(np.isfinite(x).all() for x in (imagined, actual, goal)):
        raise ValueError('trajectory latents and goal must be finite and nonempty')
    return dict(latent_mse=np.square(imagined-actual).mean(-1),
                imagined_goal_distance_sq=np.square(imagined-goal).sum(-1),
                actual_goal_distance_sq=np.square(actual-goal).sum(-1))
