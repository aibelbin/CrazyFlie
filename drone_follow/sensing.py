"""Range sensing: a fan of ray casts around the drone, via PyBullet's ray tester."""
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
