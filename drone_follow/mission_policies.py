"""Avoiders for mission legs, and the supervisor that mimics ArduPilot's AUTO -> GUIDED handover.

Every policy maps the track-frame observation (see mission_env.py) to a track-frame velocity action in
[-1, 1]^3, where 1 = MISSION_SPEED_LIMIT. Stateful ones have `reset()`, called at the start of each leg.
"""
import numpy as np

from drone_follow import config as C
from drone_follow.mission_env import M_RANGES, M_RAY_DIRS

CRUISE = C.MISSION_CRUISE / C.MISSION_SPEED_LIMIT                  # action magnitude for cruise speed
ELEVS = np.array(C.MISSION_ELEVS_DEG)
N_AZ = C.MISSION_N_AZ
RAY_ELEV = np.repeat(ELEVS, N_AZ)                                   # elevation of every ray (deg)
RAY_AZ = np.tile((np.arange(N_AZ) * 360.0 / N_AZ + 180) % 360 - 180, len(ELEVS))   # -180..180, 0 = at the waypoint


def goal_vector(obs: np.ndarray) -> np.ndarray:
    """Track-frame vector from the drone to the waypoint (m)."""
    return np.array([obs[0] * 100.0, 0.0, obs[1] * 10.0])


def ranges(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs[M_RANGES], float) * C.MISSION_RANGE


def straight_line(obs: np.ndarray) -> np.ndarray:
    """ArduPilot AUTO without avoidance: fly at the waypoint at cruise speed."""
    g = goal_vector(obs)
    n = float(np.linalg.norm(g))
    return CRUISE * g / n if n > 1e-6 else np.zeros(3)


def potential_field(obs: np.ndarray, influence: float = 6.0, gain: float = 1.5) -> np.ndarray:
    """Attraction to the waypoint plus a push away from every ray closer than `influence` metres.
    Downward rays are ignored so level ground does not lift the drone."""
    push = np.clip((influence - ranges(obs)) / influence, 0.0, 1.0) ** 2
    push[RAY_ELEV < 0] = 0.0
    a = straight_line(obs) - gain * (push[:, None] * M_RAY_DIRS).sum(axis=0)
    n = float(np.linalg.norm(a))
    return a / n if n > 1.0 else a


class BendyRuler:
    """ArduPilot's BendyRuler, simplified to the sensor rays. Every ray not pointing steeply down is a candidate
    direction; it is open if it and its two azimuth neighbours (so the gap fits the drone) see at least
    `lookahead + margin` metres (13 m by default, just above the Supervisor's 12 m trigger). Fly the open one
    closest to the waypoint direction, keeping the previous choice unless another is more than `hysteresis_deg`
    closer, so it does not flip between two gaps."""

    def __init__(self, lookahead: float = 10.0, hysteresis_deg: float = 10.0, min_elev: float = -15.0,
                 margin: float = 3.0):
        self.lookahead, self.hysteresis_deg, self.min_elev, self.margin = lookahead, hysteresis_deg, min_elev, margin
        self.reset()

    def reset(self):
        self._last = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        r = ranges(obs).reshape(len(ELEVS), N_AZ)
        free = np.minimum(r, np.minimum(np.roll(r, 1, axis=1), np.roll(r, -1, axis=1))).flatten()
        allowed = RAY_ELEV >= self.min_elev
        open_ = allowed & (free >= self.lookahead + self.margin)
        if not open_.any():                                          # boxed in: creep toward the most open direction
            i = int(np.argmax(np.where(allowed, free, -1.0)))
            self._last = None
            return 0.3 * M_RAY_DIRS[i]
        g = goal_vector(obs)
        cos = M_RAY_DIRS @ (g / (np.linalg.norm(g) + 1e-9))
        angle = np.where(open_, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))), np.inf)
        best = int(np.argmin(angle))
        if self._last is not None and open_[self._last] and angle[self._last] <= angle[best] + self.hysteresis_deg:
            best = self._last
        self._last = best
        if angle[best] < 360.0 / N_AZ / 2:                           # the waypoint direction itself is open
            return straight_line(obs)
        return CRUISE * M_RAY_DIRS[best]


class Supervisor:
    """ArduPilot-style handover. AUTO: fly straight at the waypoint. Switch to GUIDED (the avoider) when any ray
    within `cone_deg` of the waypoint direction (and not pointing steeply down) reads under `trigger` metres.
    Hand back to AUTO once the rays straight ahead (|azimuth| <= 22.5 deg, elevation -15 or 0) have all read over
    `trigger + margin` for `hold_s` seconds."""

    def __init__(self, avoider, trigger: float = 12.0, margin: float = 3.0, hold_s: float = 0.5,
                 cone_deg: float = 45.0):
        self.avoider, self.trigger, self.margin = avoider, trigger, margin
        self.hold = max(1, int(round(hold_s * C.CTRL_FREQ)))
        self.cone = (np.abs(RAY_AZ) <= cone_deg) & (RAY_ELEV >= -15.0)
        self.ahead = (np.abs(RAY_AZ) <= 22.5) & np.isin(RAY_ELEV, (-15.0, 0.0))
        self.reset()

    def reset(self):
        self.guided, self._clear = False, 0
        if hasattr(self.avoider, "reset"):
            self.avoider.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        r = ranges(obs)
        if not self.guided:
            if (r[self.cone] < self.trigger).any():
                self.guided, self._clear = True, 0
                if hasattr(self.avoider, "reset"):
                    self.avoider.reset()
        else:
            self._clear = self._clear + 1 if (r[self.ahead] > self.trigger + self.margin).all() else 0
            if self._clear >= self.hold:
                self.guided = False
        return np.asarray(self.avoider(obs), float).reshape(3) if self.guided else straight_line(obs)


def load_ppo(path: str):
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]
