"""Range sensing: a fan of ray casts around the drone, via PyBullet's ray tester."""
import numpy as np
import pybullet as p

from drone_follow import config as C


def _ray_directions() -> np.ndarray:
    dirs = []
    for el in np.radians(C.ELEVS_DEG):
        for k in range(C.N_AZ):
            az = 2 * np.pi * k / C.N_AZ
            dirs.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.array(dirs)


RAY_DIRS = _ray_directions()
N_RAYS = len(RAY_DIRS)


def range_scan(client: int, pos: np.ndarray) -> np.ndarray:
    """Distance to the nearest surface along each ray, capped at MAX_RANGE."""
    starts = pos + RAY_DIRS * C.RAY_START_OFFSET
    ends = pos + RAY_DIRS * C.MAX_RANGE
    hits = p.rayTestBatch(starts.tolist(), ends.tolist(), physicsClientId=client)
    frac = np.fromiter((h[2] for h in hits), dtype=float, count=N_RAYS)
    return C.RAY_START_OFFSET + frac * (C.MAX_RANGE - C.RAY_START_OFFSET)
