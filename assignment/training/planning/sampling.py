"""Precommitted G6 stopping rule; no state, RNG, models or simulator inputs."""


def improvement(distance_now, best_cost):
    return (distance_now-best_cost)/max(distance_now, 1e-8)


def stop_reason(distance_now, best_cost, threshold, count, maximum=48, minimum=.20):
    # At 48, search ends unconditionally; trace still records achieved cost and r.
    if count >= maximum:
        return 'budget_exhausted'
    if distance_now <= 1e-8:
        return 'near_zero_distance'
    if best_cost < threshold:
        return 'predicted_threshold'
    if improvement(distance_now, best_cost) >= minimum:
        return 'sufficient_improvement'
    return None
