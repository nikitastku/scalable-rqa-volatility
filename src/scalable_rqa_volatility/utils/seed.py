"""
Set global random seeds for reproducible experiments.

This module provides a helper for synchronizing randomness across Python hash
seeding, the standard ``random`` module, and NumPy. It is used by preprocessing,
training, and evaluation scripts to make repeated runs more deterministic.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set seed across common libraries for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)