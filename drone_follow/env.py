"""Gymnasium environment: follow a walking person while avoiding debris and rival drones.

Built on gym-pybullet-drones (Crazyflie quadrotor physics + its PID flight controller).
The policy outputs a 3D velocity command; the library's controller turns it into
motor RPMs.

Observation (OBS_DIM floats, all roughly in [-1, 1]):
    0:3    unit vector drone -> follow point (world frame)
    3      distance to follow point / 8
    4:7    person velocity (m/s)
    7:10   drone velocity / SPEED_LIMIT
    10:12  roll, pitch
    12     drone altitude / 3
    13:16  previous action
    16:40  range strip (N_RAYS), metres / MAX_RANGE
    40:64  range strip rate of change (closing speed / 2 m/s), clipped
"""
from collections import deque

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics

from drone_follow import config as C
from drone_follow.f450 import F450, F450Mixin, WindField
from drone_follow.scenario import Scenario
from drone_follow.sensing import N_RAYS, RangeSensor, range_scan

OBS_DIM = 16 + 2 * N_RAYS
RANGE_SLICE = slice(16, 16 + N_RAYS)


class FollowAvoidAviary(F450Mixin, BaseRLAviary):
    def __init__(self, gui: bool = False, difficulty: int = 2,
                 episode_sec: float = C.EPISODE_SEC, seed: int | None = None,
                 pedestrians: bool = True, realistic: bool = False):
        self.difficulty = difficulty
        self.pedestrians = pedestrians
        self.realistic = realistic          # walking figures + scenery (visual only)
        self.episode_sec = episode_sec
        self._rng = np.random.default_rng(seed)
        self._scenario = Scenario(self._rng, difficulty, pedestrians, realistic)
        self._prev_action = np.zeros(3)
        self._ranges = np.full(N_RAYS, C.MAX_RANGE)
        self._prev_ranges = self._ranges.copy()
        self._clearance = np.inf
        self._crashed = False
        self._first_obs = True
        self._tilt = 0.0
        self._agl = C.FOLLOW_ALT
        self._reset_sensing()
        super().__init__(
            drone_model=F450 if C.F450 else DroneModel.CF2X, num_drones=1,
            initial_xyzs=self._scenario.drone_start[None, :].copy(),
            physics=Physics.PYB, pyb_freq=C.PYB_FREQ, ctrl_freq=C.CTRL_FREQ,
            gui=gui, record=False, obs=ObservationType.KIN, act=ActionType.VEL)
        self.SPEED_LIMIT = C.SPEED_LIMIT     # the library default (0.25 m/s) is far too slow
        if C.F450:
            self._f450_setup(None)

    # ----------------------------------------------------------------- spaces
    def _actionSpace(self):
        return spaces.Box(-1.0, 1.0, shape=(1, 3), dtype=np.float32)

    def _observationSpace(self):
        return spaces.Box(-10.0, 10.0, shape=(1, OBS_DIM), dtype=np.float32)

    def _preprocessAction(self, action):
        a = np.clip(np.asarray(action, float).reshape(3), -1.0, 1.0)
        mag = min(float(np.linalg.norm(a)), 1.0)
        if C.F450:                                  # the F450 controller runs at the physics rate (see f450.py)
            n = float(np.linalg.norm(a))
            self._v_target = self.SPEED_LIMIT * mag * (a / n if n > 0 else a)
            return np.full((1, 4), self.HOVER_RPM)
        return super()._preprocessAction(np.array([[*a, mag]]))   # (direction, speed)

    # ------------------------------------------------------------- lifecycle
    def set_difficulty(self, difficulty: int):
        self.difficulty = difficulty          # applies from the next reset

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._scenario = Scenario(self._rng, self.difficulty, self.pedestrians, self.realistic)
        self.INIT_XYZS = self._scenario.drone_start[None, :].copy()
        self._prev_action = np.zeros(3)
        self._ranges = np.full(N_RAYS, C.MAX_RANGE)
        self._prev_ranges = self._ranges.copy()
        self._crashed = False
        self._first_obs = True
        self._f450_reset()
        self._reset_sensing()
        if C.F450:
            self._f450_set_wind(self._wind_for_level())
        return super().reset(seed=seed, options=options)

    def _reset_sensing(self):
        self._sense_ranges = RangeSensor(self._rng, C.MAX_RANGE)
        self._rel_hist = deque(maxlen=C.SENSOR_DELAY + 1)
        self._act_queue = deque([np.zeros(3)] * C.ACT_DELAY, maxlen=max(C.ACT_DELAY, 1))

    def _wind_for_level(self):
        """Levels 4-5 are windy (F450 only): 4 = breeze with light gusts, 5 = strong wind with gusts.
        Lower levels use the DF_WIND / DF_GUST settings (calm by default)."""
        lvl, rng = self.difficulty, self._rng
        if lvl == 4:
            return WindField(rng, rng.uniform(0.0, 3.0), 1.0)
        if lvl >= 5:
            return WindField(rng, rng.uniform(2.0, 6.0), 2.5)
        return WindField(rng, C.WIND_MEAN, C.WIND_GUST) if (C.WIND_MEAN or C.WIND_GUST) else None

    def _addObstacles(self):               # called from _housekeeping() after every reset
        self._scenario.attach(self.CLIENT)
        self._halo = None
        if not self.realistic:                # the live demo draws a real quadcopter model instead
            # Visual-only halo so the 9 cm quadrotor is easy to spot in rendered clips.
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.14, rgbaColor=(0.1, 0.4, 1.0, 0.8),
                                      physicsClientId=self.CLIENT)
            self._halo = p.createMultiBody(0, -1, vis, basePosition=self.INIT_XYZS[0].tolist(),
                                           physicsClientId=self.CLIENT)

    def step(self, action):
        a = np.clip(np.asarray(action, float).reshape(3), -1.0, 1.0)
        if C.ACTION_RATE_LIMIT > 0:          # smooth by construction: bound how fast the command may change
            lim = C.ACTION_RATE_LIMIT
            a = self._prev_action + np.clip(a - self._prev_action, -lim, lim)
        self._scenario.advance(self.CTRL_TIMESTEP, self.pos[0], self.vel[0])
        self._action = a
        applied = a
        if C.ACT_DELAY > 0:                  # the autopilot acts on the command from ACT_DELAY steps ago
            applied = self._act_queue.popleft()
            self._act_queue.append(a)
        out = super().step(applied)
        self._prev_action = a
        if self._halo is not None:
            p.resetBasePositionAndOrientation(self._halo, self.pos[0].tolist(), [0, 0, 0, 1],
                                              physicsClientId=self.CLIENT)
        return out

    # ----------------------------------------------------------- observation
    def _computeObs(self):
        s = self._getDroneStateVector(0)
        pos, rpy, vel = s[0:3], s[7:10], s[10:13]
        new_ranges = range_scan(self.CLIENT, pos)
        if C.NOISY:
            new_ranges = self._sense_ranges(new_ranges)
        self._prev_ranges = new_ranges if self._first_obs else self._ranges
        self._ranges = new_ranges
        self._first_obs = False
        self._clearance = self._scenario.clearance(pos)
        self._tilt = float(max(abs(rpy[0]), abs(rpy[1])))
        agl = pos[2] - self._scenario.ground(pos[:2])                    # height above the terrain
        self._agl = float(agl)
        self._crashed = (self._clearance < 0.0 or agl < C.GROUND_Z
                         or abs(rpy[0]) > C.MAX_TILT or abs(rpy[1]) > C.MAX_TILT)

        rel = self._scenario.target_point - pos
        dist = float(np.linalg.norm(rel))                 # true distance: used for the reward only
        person_vel, vel_obs, rpy_obs, rel_obs = self._scenario.person_vel, vel, rpy, rel
        if C.NOISY:                                       # what the detector, depth camera and odometry report
            self._rel_hist.append(rel)
            rng = self._rng
            rel_obs = self._rel_hist[0] + rng.standard_normal(3) * C.TARGET_NOISE
            person_vel = person_vel + rng.standard_normal(2) * C.TARGET_VEL_NOISE
            vel_obs = vel + rng.standard_normal(3) * C.STATE_VEL_NOISE
            rpy_obs = rpy + rng.standard_normal(3) * C.STATE_ATT_NOISE
        dist_obs = float(np.linalg.norm(rel_obs))
        rate = np.clip((self._ranges - self._prev_ranges) / self.CTRL_TIMESTEP / 2.0, -1.5, 1.5)
        obs = np.concatenate([
            rel_obs / (dist_obs + 1e-9), [dist_obs / 8.0],
            person_vel, [0.0], vel_obs / C.SPEED_LIMIT,
            rpy_obs[:2], [agl / 3.0], self._prev_action,
            self._ranges / C.MAX_RANGE, rate,
        ]).astype(np.float32)
        self._follow_dist = dist
        return obs[None, :]

    # ---------------------------------------------------------------- reward
    def _hazard_terms(self):
        """(collision-course urgency, nearness) in [0, 1], from the simulator's true hazard state.

        Used only to shape the training reward; the policy itself sees just the range strip.
        """
        pos, vel = self.pos[0], self.vel[0]
        urgency, min_gap = self._scenario.static_hazard(pos, vel)
        for o in self._scenario.live_obstacles():
            rel = o.pos - pos
            dist = float(np.linalg.norm(rel)) + 1e-9
            gap = dist - o.radius - C.DRONE_RADIUS
            min_gap = min(min_gap, gap)
            closing = -float(np.dot(rel / dist, o.vel - vel))      # > 0 when approaching
            if closing > 0.05:
                ttc = max(gap, 0.0) / closing
                urgency = max(urgency, float(np.clip((C.TTC_START - ttc) / C.TTC_START, 0.0, 1.0)) ** 2)
        nearness = 0.0
        if np.isfinite(min_gap):
            nearness = float(np.clip((C.CLEARANCE_START - min_gap) / C.CLEARANCE_START, 0.0, 1.0)) ** 2
        return urgency, nearness

    def _computeReward(self):
        if self._crashed:
            return C.R_COLLISION * C.REWARD_SCALE
        follow = C.R_FOLLOW * (1.0 - min(abs(self._follow_dist - C.FOLLOW_DIST), 4.0) / 4.0)
        near = float(self._ranges.min())
        proximity = -C.R_PROXIMITY * np.clip((C.PROXIMITY_START - near) / 0.7, 0.0, 1.0) ** 2
        urgency, nearness = self._hazard_terms()
        smooth = -C.R_SMOOTH * float(np.linalg.norm(self._action - self._prev_action))
        total = follow + proximity - C.R_TTC * urgency - C.R_CLEAR * nearness + smooth
        if C.R_TILT > 0:
            lean = np.clip((self._tilt - C.TILT_START) / (C.MAX_TILT - C.TILT_START), 0.0, 1.0)
            total -= C.R_TILT * lean ** 2
        return total * C.REWARD_SCALE

    def _computeTerminated(self):
        return bool(self._crashed)

    def _computeTruncated(self):
        pos = self._getDroneStateVector(0)[0:3]
        out = (np.abs(pos[:2]).max() > self._scenario.bound
               or pos[2] - self._scenario.ground(pos[:2]) > self._scenario.z_cap)
        timeout = self.step_counter / self.PYB_FREQ > self.episode_sec
        return bool(out or timeout)

    def _collision_kind(self):
        """What ended the episode: the kind of the nearest hazard, 'ground' or 'flip' (None if no crash)."""
        if not self._crashed:
            return None
        pos = self.pos[0]
        if self._clearance < 0.0:
            live = self._scenario.live_obstacles()
            best_gap, best_kind = self._scenario.nearest_static_kind(pos)[::-1]
            for o in live:
                gap = float(np.linalg.norm(pos - o.pos)) - o.radius
                if gap < best_gap:
                    best_gap, best_kind = gap, o.kind
            return best_kind
        return "ground" if self._agl < C.GROUND_Z else "flip"

    def _computeInfo(self):
        return {"collision": bool(self._crashed), "collision_kind": self._collision_kind(),
                "follow_dist": self._follow_dist,
                "min_range": float(self._ranges.min()),
                "clearance": float(self._clearance)}


class SingleDroneWrapper(gym.Wrapper):
    """Drops the library's leading num_drones axis so SB3 sees flat vectors."""

    def __init__(self, env: FollowAvoidAviary):
        super().__init__(env)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return obs[0], info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs[0], float(r), term, trunc, info

    def set_difficulty(self, difficulty: int):
        self.env.set_difficulty(difficulty)


def make_env(difficulty: int = 2, seed: int | None = None, gui: bool = False,
             episode_sec: float = C.EPISODE_SEC, pedestrians: bool = True,
             realistic: bool = False, spectate: dict | None = None):
    """Thunk that builds an env. `spectate` = dict(path=..., label=...) makes it write live poses for the 3D viewer."""
    def _thunk():
        env = SingleDroneWrapper(FollowAvoidAviary(
            gui=gui, difficulty=difficulty, seed=seed, episode_sec=episode_sec,
            pedestrians=pedestrians, realistic=realistic))
        if spectate:
            from drone_follow.spectator import SpectatorWrapper
            env = SpectatorWrapper(env, **spectate)
        return env
    return _thunk
