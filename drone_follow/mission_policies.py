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
    """A classic 2D potential field: attraction to the waypoint plus a horizontal push away from every ray
    closer than `influence` metres. Downward rays are ignored so level ground does not lift the drone, and the
    push has no vertical part (rays tilted up would otherwise press it toward the ground): it goes around
    obstacles and never climbs over them."""
    push = np.clip((influence - ranges(obs)) / influence, 0.0, 1.0) ** 2
    push[RAY_ELEV < 0] = 0.0
    rep = (push[:, None] * M_RAY_DIRS).sum(axis=0)
    rep[2] = 0.0
    a = straight_line(obs) - gain * rep
    n = float(np.linalg.norm(a))
    return a / n if n > 1.0 else a


class BendyRuler:
    """ArduPilot's BendyRuler, simplified to the sensor rays. Every ray not pointing steeply down is a candidate
    direction; it is open if it, its two azimuth neighbours (so the gap fits the drone) and the sector directly
    below it (same azimuth, next lower elevation ring; the lowest ring has none) all see at least
    `lookahead + margin` metres (13 m by default, just above the Supervisor's 12 m trigger). The sector below is
    the vertical clearance: level flight just over a canopy or roof counts as blocked, so it climbs higher before
    crossing. Fly the open one closest to the waypoint direction, keeping the previous choice unless another is
    more than `hysteresis_deg` closer, so it does not flip between two gaps."""

    def __init__(self, lookahead: float = 10.0, hysteresis_deg: float = 10.0, min_elev: float = -15.0,
                 margin: float = 3.0):
        self.lookahead, self.hysteresis_deg, self.min_elev, self.margin = lookahead, hysteresis_deg, min_elev, margin
        self.reset()

    def reset(self):
        self._last = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        r = ranges(obs).reshape(len(ELEVS), N_AZ)
        side = np.minimum(r, np.minimum(np.roll(r, 1, axis=1), np.roll(r, -1, axis=1)))
        below = np.vstack([np.full((1, N_AZ), np.inf), r[:-1]])     # same azimuth, one ring down
        free = np.minimum(side, below).flatten()
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


CONE_ELEVS = ELEVS[ELEVS >= 0][:2]          # the level ring and the first one up (0 and 15 deg)
LOW_ELEV = ELEVS[ELEVS < 0].max()           # the first ring down (-15 deg)
SIN_LOW = float(np.sin(np.radians(-LOW_ELEV)))   # flat ground `agl` below reads agl / SIN_LOW in that ring


class Supervisor:
    """ArduPilot-style handover. AUTO: fly straight at the waypoint. Switch to GUIDED (the avoider) when
    - any sector of the trigger cone reads under `trigger` metres. The cone is |azimuth| <= `cone_deg` at the
      level and first upward elevations (CONE_ELEVS: 0 and 15 deg); or
    - any first-down sector (LOW_ELEV, -15 deg) within |azimuth| <= `cone_deg` reads under
      min(trigger, ground_trigger * agl / SIN_LOW), agl = obs[7] * 10. Flat ground reads agl / SIN_LOW there, so
      the ground itself never triggers, but a canopy or roof close under the flight path does.
    Hand back to AUTO once both have read over the clear thresholds (`trigger + margin`, and
    min(trigger + margin, ground_clear * agl / SIN_LOW)) for `hold_s` seconds: the same masks for both, so it
    does not re-trigger at once. In AUTO, while any downward sector within the cone azimuths reads under
    `floor_m`, the command never descends (vertical component >= 0), so AUTO does not settle onto a roof it has
    just climbed over on its way back to the waypoint's altitude."""

    def __init__(self, avoider, trigger: float = 12.0, margin: float = 3.0, hold_s: float = 0.5,
                 cone_deg: float = 45.0, ground_trigger: float = 0.6, ground_clear: float = 0.7,
                 floor_m: float = 3.0):
        self.avoider, self.trigger, self.margin = avoider, trigger, margin
        self.ground_trigger, self.ground_clear, self.floor_m = ground_trigger, ground_clear, floor_m
        self.hold = max(1, int(round(hold_s * C.CTRL_FREQ)))
        in_az = np.abs(RAY_AZ) <= cone_deg
        self.cone = in_az & np.isin(RAY_ELEV, CONE_ELEVS)
        self.low = in_az & (RAY_ELEV == LOW_ELEV)
        self.below = in_az & (RAY_ELEV < 0)
        self.reset()

    def reset(self):
        self.guided, self._clear = False, 0
        if hasattr(self.avoider, "reset"):
            self.avoider.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        r = ranges(obs)
        ground = max(float(obs[7]) * 10.0, 0.0) / SIN_LOW               # what flat ground reads in the low ring
        if not self.guided:
            if ((r[self.cone] < self.trigger).any()
                    or (r[self.low] < min(self.trigger, self.ground_trigger * ground)).any()):
                self.guided, self._clear = True, 0
                if hasattr(self.avoider, "reset"):
                    self.avoider.reset()
        else:
            clear = ((r[self.cone] > self.trigger + self.margin).all()
                     and (r[self.low] > min(self.trigger + self.margin, self.ground_clear * ground)).all())
            self._clear = self._clear + 1 if clear else 0
            if self._clear >= self.hold:
                self.guided = False
        if self.guided:
            return np.asarray(self.avoider(obs), float).reshape(3)
        a = straight_line(obs)
        if (r[self.below] < self.floor_m).any():                     # something close underneath: do not descend
            a[2] = max(a[2], 0.0)
        return a


def load_ppo(path: str):
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]
