"""Hand-written baselines the trained policy is compared against.

Both take the flat observation (see env.py) and return a velocity action in [-1, 1]^3.
"""
import numpy as np

from drone_follow import config as C
from drone_follow.env import RANGE_SLICE
from drone_follow.sensing import RAY_DIRS


def naive_follow(obs: np.ndarray) -> np.ndarray:
    """Fly straight at the person; ignores obstacles entirely."""
    direction, dist = obs[0:3], obs[3] * 8.0
    return direction * np.clip((dist - C.FOLLOW_DIST) / 1.5, -1.0, 1.0)


def reactive_follow(obs: np.ndarray, influence: float = 2.0, gain: float = 2.5) -> np.ndarray:
    """Follow plus a potential-field push away from anything the range strip sees."""
    ranges = obs[RANGE_SLICE] * C.MAX_RANGE
    close = np.clip((influence - ranges) / influence, 0.0, 1.0) ** 2
    repulse = -(close[:, None] * RAY_DIRS).sum(axis=0)
    a = naive_follow(obs) + gain * repulse / len(ranges) * 4.0
    n = np.linalg.norm(a)
    return a / n if n > 1.0 else a
