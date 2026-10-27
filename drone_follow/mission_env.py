"""Gymnasium environment: fly one mission leg to a waypoint at cruise speed, avoiding static obstacles.

Always the F450 (ArduPilot-style velocity controller, motor lag, wind) with sensor noise and a one-step
command delay, whatever DF_DRONE says: this is the drone the missions will fly.

Everything the policy sees and commands is in the *track frame*: x points horizontally at the waypoint,
z is up, like a drone that yaws to face its next waypoint (ArduPilot's default). Rays are cast in that frame.

Observation (M_OBS_DIM floats):
    0        horizontal distance to the waypoint / 100, capped at 1.5
    1        height of the waypoint above the drone / 10
    2:5      drone velocity (track frame) / MISSION_SPEED_LIMIT
    5:7      roll, pitch
    7        height above the ground / 10
    8:11     previous action (track frame)
    11:91    ranges / MISSION_RANGE (elevation-major: index = e * MISSION_N_AZ + k, k = 0 straight ahead)
    91:171   range rate of change (closing speed / 5 m/s), clipped to [-1.5, 1.5]
"""
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType, Physics

from drone_follow import config as C
from drone_follow.f450 import F450, F450Mixin, WindField
from drone_follow.mission_world import Leg, attach_leg, make_leg
from drone_follow.sensing import RangeSensor, cast, ray_directions

M_RAY_DIRS = ray_directions(C.MISSION_N_AZ, C.MISSION_ELEVS_DEG)
M_N_RAYS = len(M_RAY_DIRS)
M_OBS_DIM = 11 + 2 * M_N_RAYS
M_PREV_ACTION = slice(8, 11)
M_RANGES = slice(11, 11 + M_N_RAYS)
LEVEL_OR_UP = np.repeat(np.array(C.MISSION_ELEVS_DEG) >= 0, C.MISSION_N_AZ)     # rays that do not look down


def yaw_matrix(yaw: float) -> np.ndarray:
    """Track frame -> world frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class MissionAviary(F450Mixin, BaseRLAviary):
    def __init__(self, level: int = 1, seed: int | None = None, gui: bool = False, length: float | None = None):
        self.level, self.length = level, length
        self._rng = np.random.default_rng(seed)
        self._leg = make_leg(self._rng, level, length)
        self._reset_state()
        super().__init__(
            drone_model=F450, num_drones=1, initial_xyzs=self._leg.start[None, :].copy(),
            physics=Physics.PYB, pyb_freq=C.PYB_FREQ, ctrl_freq=C.CTRL_FREQ,
            gui=gui, record=False, obs=ObservationType.KIN, act=ActionType.VEL)
        self.SPEED_LIMIT = C.MISSION_SPEED_LIMIT
        self._f450_setup(None)

    # ----------------------------------------------------------------- spaces
    def _actionSpace(self):
        return spaces.Box(-1.0, 1.0, shape=(1, 3), dtype=np.float32)

    def _observationSpace(self):
        return spaces.Box(-10.0, 10.0, shape=(1, M_OBS_DIM), dtype=np.float32)

    def _preprocessAction(self, action):
        a = np.clip(np.asarray(action, float).reshape(3), -1.0, 1.0)
        n = float(np.linalg.norm(a))
        if n > 1.0:
            a = a / n
        self._v_target = self.SPEED_LIMIT * (yaw_matrix(self._yaw) @ a)
        return np.full((1, 4), self.HOVER_RPM)

    # ------------------------------------------------------------- lifecycle
    def set_level(self, level: int):
        self.level = level                    # applies from the next reset

    @property
    def leg(self) -> Leg:
        return self._leg

    def _reset_state(self):
        self._yaw = 0.0
        self._action = self._prev_action = np.zeros(3)
        self._ranges = np.full(M_N_RAYS, C.MISSION_RANGE)
        self._prev_ranges = self._ranges.copy()
        self._first_obs = True
        self._crashed = self._success = False
        self._clearance = np.inf
        self._agl = C.MISSION_ALT
        self._dist = self._prev_dist = None
        self._path, self._last_pos = 0.0, None
        self._sense = RangeSensor(self._rng, C.MISSION_RANGE)
        self._act_queue = deque([np.zeros(3)] * C.MISSION_ACT_DELAY, maxlen=C.MISSION_ACT_DELAY)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._leg = make_leg(self._rng, self.level, self.length)
        self.INIT_XYZS = self._leg.start[None, :].copy()
        self._reset_state()
        self._f450_reset()
        self._f450_set_wind(WindField(self._rng, self._rng.uniform(0.0, 2.0 + self.level), 0.5 + 0.5 * self.level))
        return super().reset(seed=seed, options=options)

    def _addObstacles(self):                 # called from _housekeeping() after every reset
        attach_leg(self._leg, self.CLIENT)

    def step(self, action):
        a = np.clip(np.asarray(action, float).reshape(3), -1.0, 1.0)
        self._action = a
        self._act_queue.append(a)            # the autopilot acts on the command from a step ago
        out = super().step(self._act_queue.popleft())
        self._prev_action = a
        return out

    # ----------------------------------------------------------- observation
    def _computeObs(self):
        s = self._getDroneStateVector(0)
        pos, rpy, vel = s[0:3], s[7:10], s[10:13]
        leg = self._leg
        to_goal = leg.goal - pos
        if np.linalg.norm(to_goal[:2]) > 0.5:      # right at the waypoint the heading is undefined: keep the last
            self._yaw = float(np.arctan2(to_goal[1], to_goal[0]))
        R = yaw_matrix(self._yaw)
        ranges = self._sense(cast(self.CLIENT, pos, M_RAY_DIRS @ R.T, C.MISSION_RANGE, C.MISSION_RAY_START))
        self._prev_ranges = ranges if self._first_obs else self._ranges
        self._ranges, self._first_obs = ranges, False

        self._clearance = leg.clearance(pos)
        self._agl = float(pos[2] - leg.ground(pos[:2]))
        self._crashed = bool(self._clearance < 0.0 or self._agl < C.GROUND_Z
                             or abs(rpy[0]) > C.MAX_TILT or abs(rpy[1]) > C.MAX_TILT)
        self._prev_dist, self._dist = self._dist, float(np.linalg.norm(to_goal))
        self._success = bool(self._dist < C.MISSION_GOAL_RADIUS and not self._crashed)
        if self._last_pos is not None:
            self._path += float(np.linalg.norm(pos - self._last_pos))
        self._last_pos = pos.copy()

        rng = self._rng
        vel_obs = vel + rng.standard_normal(3) * C.STATE_VEL_NOISE
        rpy_obs = rpy + rng.standard_normal(3) * C.STATE_ATT_NOISE
        rate = np.clip((self._ranges - self._prev_ranges) / self.CTRL_TIMESTEP / 5.0, -1.5, 1.5)
        obs = np.concatenate([
            [min(float(np.linalg.norm(to_goal[:2])) / 100.0, 1.5), to_goal[2] / 10.0],
            (R.T @ vel_obs) / C.MISSION_SPEED_LIMIT, rpy_obs[:2], [self._agl / 10.0],
            self._prev_action, self._ranges / C.MISSION_RANGE, rate,
        ]).astype(np.float32)
        return obs[None, :]

    # ---------------------------------------------------------------- outcome
    def _elapsed(self) -> float:
        return self.step_counter / self.PYB_FREQ

    def _failure(self):
        """'out_of_bounds', 'timeout' or None."""
        pos = self.pos[0]
        if (abs(pos[1]) > C.MISSION_HALF_WIDTH or pos[0] < -15.0 or pos[0] > self._leg.length + 15.0
                or self._agl > C.MISSION_CEILING):
            return "out_of_bounds"
        if self._elapsed() > C.MISSION_TIME_FACTOR * self._leg.length / C.MISSION_CRUISE:
            return "timeout"
        return None

    def _computeReward(self):
        if self._crashed:
            return C.MR_CRASH
        if self._success:
            return C.MR_SUCCESS
        r = 0.0
        if self._prev_dist is not None:
            r += C.MR_PROGRESS * (self._prev_dist - self._dist)
        near = float(self._ranges[LEVEL_OR_UP].min())
        r -= C.MR_PROXIMITY * float(np.clip((C.MR_PROXIMITY_START - near) / C.MR_PROXIMITY_START, 0.0, 1.0)) ** 2
        r -= C.MR_SMOOTH * float(np.linalg.norm(self._action - self._prev_action))
        if self._failure():
            r += C.MR_FAIL
        return r

    def _computeTerminated(self):
        return bool(self._crashed or self._success)

    def _computeTruncated(self):
        return bool(not self._crashed and not self._success and self._failure())

    def _collision_kind(self):
        if not self._crashed:
            return None
        if self._clearance < 0.0:
            return self._leg.nearest_kind(self.pos[0])
        return "ground" if self._agl < C.GROUND_Z else "flip"

    def _computeInfo(self):
        done = self._crashed or self._success
        return {"collision": bool(self._crashed), "collision_kind": self._collision_kind(),
                "success": bool(self._success), "failure": None if done else self._failure(),
                "dist_to_goal": self._dist, "path_m": self._path,
                "leg_m": float(np.linalg.norm(self._leg.goal - self._leg.start)),
                "time_s": self._elapsed(), "min_range": float(self._ranges.min())}


class MissionWrapper(gym.Wrapper):
    """Drops the library's leading num_drones axis so SB3 sees flat vectors."""

    def __init__(self, env: MissionAviary):
        super().__init__(env)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(M_OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return obs[0], info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs[0], float(r), term, trunc, info

    def set_level(self, level: int):
        self.env.set_level(level)


def make_mission_env(level: int = 1, seed: int | None = None, length: float | None = None, gui: bool = False):
    """Thunk that builds a mission env (SubprocVecEnv wants thunks)."""
    def _thunk():
        return MissionWrapper(MissionAviary(level=level, seed=seed, gui=gui, length=length))
    return _thunk
