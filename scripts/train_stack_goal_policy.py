#!/usr/bin/env python3
"""Compatibility wrapper for the v2 offline goal-conditioned actor finetuner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assignment.training.training.train_goal_policy import main  # noqa: E402

if __name__ == '__main__':
    main()
