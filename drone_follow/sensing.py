"""Range sensing: a fan of ray casts around the drone, via PyBullet's ray tester."""
from collections import deque

import numpy as np
import pybullet as p

from drone_follow import config as C


def ray_directions(n_az: int, elevs_deg) -> np.ndarray:
    """Unit vectors, elevation-major: index = e * n_az + k, with k = 0 along +x and k increasing toward +y."""
    dirs = []
    for el in np.radians(elevs_deg):
        for k in range(n_az):
            az = 2 * np.pi * k / n_az
            dirs.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.array(dirs)


RAY_DIRS = ray_directions(C.N_AZ, C.ELEVS_DEG)
N_RAYS = len(RAY_DIRS)


def cast(client: int, pos: np.ndarray, dirs: np.ndarray, max_range: float,
         start_offset: float = C.RAY_START_OFFSET) -> np.ndarray:
    """Distance to the nearest surface along each direction, capped at `max_range`."""
    starts = pos + dirs * start_offset
    ends = pos + dirs * max_range
    hits = p.rayTestBatch(starts.tolist(), ends.tolist(), physicsClientId=client)
    frac = np.fromiter((h[2] for h in hits), dtype=float, count=len(dirs))
    return start_offset + frac * (max_range - start_offset)


def range_scan(client: int, pos: np.ndarray) -> np.ndarray:
    """The follow env's strip: distance to the nearest surface along each ray, capped at MAX_RANGE."""
    return cast(client, pos, RAY_DIRS, C.MAX_RANGE)


class RangeSensor:
    """Depth sensor model: delayed, noisy, occasional dropouts (read max range), lightly smoothed.
    Draws from the generator it is given, so seeded episodes stay reproducible."""

    def __init__(self, rng: np.random.Generator, max_range: float):
        self.rng, self.max_range = rng, max_range
        self._hist = deque(maxlen=C.SENSOR_DELAY + 1)            # true ranges, oldest = what the sensor reports now
        self._smooth = None

    def __call__(self, true_ranges: np.ndarray) -> np.ndarray:
        self._hist.append(true_ranges)
        r = self._hist[0].copy()
        rel_sigma, abs_sigma = C.RANGE_NOISE
        r = r + self.rng.standard_normal(r.shape) * (rel_sigma * r + abs_sigma)
        r[self.rng.random(r.shape) < C.RANGE_DROPOUT] = self.max_range
        r = np.clip(r, 0.05, self.max_range)
        self._smooth = r if self._smooth is None else 0.5 * self._smooth + 0.5 * r
        return self._smooth.copy()
