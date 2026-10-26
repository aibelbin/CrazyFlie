"""What the 3D viewer needs from a running simulation, as JSON: poses every step, scenery once per episode."""
import json

import numpy as np


def _r(v, n=3):
    return [round(float(x), n) for x in v]


def pose_snapshot(sim, episode_id: int, status: str, policy: str, difficulty: int,
                  episodes: int, collisions: int) -> str:
    """Poses of the drone, the person and every moving obstacle (sim coordinates, z up)."""
    sc = sim.scenario
    obs = [{"id": i, "k": o.kind, "p": _r(o.pos), "v": _r(o.vel), "r": round(float(o.radius), 3),
            "a": int(o.active), "g": o.group} for i, o in enumerate(sc.obstacles)]
    wind = sim._wind.mean + sim._wind.dev if sim._wind is not None else np.zeros(3)
    msg = {"ep": episode_id,
           "drone": {"p": _r(sim.pos[0]), "q": _r(sim.quat[0], 4), "v": _r(sim.vel[0]),
                     "gz": round(sc.ground(sim.pos[0][:2]), 3)},
           "person": {"p": _r(sc.person), "v": _r(sc.person_vel), "gz": round(sc.ground(sc.person), 3)},
           "wind": _r(wind, 2), "obs": obs, "status": status, "policy": policy, "difficulty": difficulty,
           "episodes": episodes, "collisions": collisions}
    return json.dumps(msg, separators=(",", ":"))


def world_snapshot(sc, episode_id: int) -> bytes:
    """Static scenery for the viewer (fetched once per episode): terrain heights, trees and rocks."""
    terrain = None
    if sc.terrain is not None:
        from drone_follow.world import GRID, HALF
        terrain = {"half": HALF, "n": GRID, "z": [round(float(v), 2) for v in sc.terrain.grid().flatten()]}
    statics = [{"k": str(k), "p": _r(pp, 2), "r": round(float(r), 2), "g": int(g)}
               for pp, r, k, g in zip(sc.static_pos, sc.static_r, sc.static_kind, sc.static_group)]
    return json.dumps({"ep": episode_id, "terrain": terrain, "statics": statics}, separators=(",", ":")).encode()
