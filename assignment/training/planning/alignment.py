"""Real-observation stagnation gate with forward (G3) or bidirectional (G4/G5) matching."""
from collections import deque
from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class AlignmentConfig:
    min_dwell_s: float = 24.
    window_s: float = 8.
    improvement_fraction: float = .10
    edge_samples: int = 4
    lookahead: int = 3
    confirmations: int = 3
    block_s: float = .5

    def __post_init__(self):
        if self.min_dwell_s < 0 or self.window_s <= 0 or not 0 < self.improvement_fraction < 1:
            raise ValueError('invalid stagnation timing/improvement')
        if min(self.edge_samples, self.lookahead, self.confirmations) < 1 or self.block_s <= 0:
            raise ValueError('invalid confirmation/window counts')
        if self.window_s/self.block_s+1 < 2*self.edge_samples:
            raise ValueError('window must hold nonoverlapping beginning/end samples')


class GoalRealigner:
    def __init__(self, config=None, bidirectional=False):
        self.config = AlignmentConfig(**(config or {}))
        self.bidirectional = bidirectional
        self.reset()

    def reset(self):
        self.tick, self.index, self.since = -1, None, 0.
        self.history = deque()
        self.pending, self.count = None, 0

    def changed(self, index, distance):
        self.index, self.since = index, self.tick*self.config.block_s
        self.history.clear()
        self.history.append((self.since, float(distance)))
        self.pending, self.count = None, 0

    def observe(self, index, distance, future_supplier):
        """Caller applies official advancement first; supplier reads only real z and goal latents."""
        if not self.bidirectional and self.index is not None and index < self.index:
            raise ValueError('goal indices must not go backwards')
        self.tick += 1
        now, cfg = self.tick*self.config.block_s, self.config
        if index != self.index:
            self.changed(index, distance)
        elif not self.history or self.history[-1][0] != now:
            self.history.append((now, float(distance)))
        while self.history and self.history[0][0] < now-cfg.window_s-1e-9:
            self.history.popleft()
        full = bool(self.history and now-self.history[0][0] >= cfg.window_s-1e-9)
        info = dict(stagnation_condition_met=False, future_match_found=None,
                    nearby_match_found=None, alignment_search_mode='bidirectional' if self.bidirectional else 'forward',
                    stagnation_start_median=None, stagnation_end_median=None,
                    stagnation_relative_improvement=None, future_goal_checks=[],
                    alignment_goal_checks=[],
                    pending_match_index=None, pending_match_count=0)
        if full:
            values = [d for _, d in self.history]
            start = statistics.median(values[:cfg.edge_samples])
            end = statistics.median(values[-cfg.edge_samples:])
            improved = start > 0 and end <= (1-cfg.improvement_fraction)*start
            info.update(stagnation_start_median=start, stagnation_end_median=end,
                        stagnation_relative_improvement=(start-end)/start if start > 0 else None)
        else:
            improved = False
        stalled = now-self.since >= cfg.min_dwell_s-1e-9 and full and not improved
        info['stagnation_condition_met'] = stalled
        if not stalled:
            self.pending, self.count = None, 0
            return None, info
        checks = []
        for candidate in future_supplier():
            j, d, threshold = candidate['index'], float(candidate['distance_sq']), float(candidate['threshold_sq'])
            valid = max(0, index-cfg.lookahead) <= j <= index+cfg.lookahead if self.bidirectional else index < j <= index+cfg.lookahead
            if not valid:
                raise ValueError('supplier exceeded the configured search range')
            q = d/threshold if threshold > 0 else None
            checks.append(dict(index=int(j), distance_sq=d, threshold_sq=threshold, q=q))
        info['future_goal_checks'] = checks
        info['alignment_goal_checks'] = checks
        matched = [c for c in checks if c['q'] is not None and math.isfinite(c['q']) and c['q'] < 1]
        info['nearby_match_found'] = bool(matched)
        info['future_match_found'] = any(c['index']>index for c in matched)
        if not matched:
            self.pending, self.count = None, 0
            return None, info
        best = min(matched, key=lambda c: (c['q'], c['index'] != index, abs(c['index']-index), c['index'])) if self.bidirectional else min(matched, key=lambda c: (c['q'], c['index']))
        info['best_match_index'] = best['index']
        if best['index'] == index:
            self.pending, self.count = None, 0
            return None, info
        self.count = self.count+1 if self.pending == best['index'] else 1
        self.pending = best['index']
        info.update(pending_match_index=self.pending, pending_match_count=self.count)
        if self.count < cfg.confirmations:
            return None, info
        info.update(matched_goal_index=best['index'], matched_goal_distance_sq=best['distance_sq'],
                    matched_goal_threshold_sq=best['threshold_sq'], matched_goal_q=best['q'],
                    matched_confirmations=self.count)
        return best['index'], info
