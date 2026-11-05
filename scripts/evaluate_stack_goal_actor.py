#!/usr/bin/env python3
"""Compatibility entry point for the v2 A1/A2/A3 evaluator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assignment.training.planning.evaluate_goal_actor import main  # noqa: E402

if __name__ == '__main__':
    main()
