"""Frozen planner variants; experiment plans choose which groups to evaluate."""

PLANNING_HORIZONS = {
    'A4-32-SHAFAA-2S': 4,
    'A4-32-SHAFAA-4S': 8,
    'A4-32-SHAFAA-8S': 16,
}
CANDIDATES = {
    'A1': 1, 'A2': 16, 'A3': 16, 'A4-16': 16, 'A4-32': 32, 'A4-32-HOLD': 32,
    'A4-16-RUIMING': 16,
    'A4-32-T1': 32, 'A4-32-T2': 32, 'A4-32-T1-L4': 32,
    'A4-32-T1-H8': 32, 'A4-32-T1-H12': 32, 'A4-32-T1-L4-HOLD': 32,
    **{mode: 32 for mode in PLANNING_HORIZONS},
    'A5-16': 16, 'A5-32': 32,
}
MODES = tuple(CANDIDATES)
TRAJECTORY_SPECS = {
    'A4-16-RUIMING': ('block16', .25),
    'A4-32-T1': ('block16', .25),
    'A4-32-T2': ('block16', .50),
    'A4-32-T1-L4': ('late4_mean', .25),
    'A4-32-T1-H8': ('block8', .25),
    'A4-32-T1-H12': ('block12', .25),
    'A4-32-T1-L4-HOLD': ('late4_mean', .25),
}
