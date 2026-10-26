"""DIY F450 quadcopter: physics model, an ArduPilot-style velocity controller, motor lag and wind.

Why this exists: the target hardware is a 1.5 kg F450 flown by ArduPilot in GUIDED mode, which accepts
velocity setpoints. The library's Crazyflie model and PID controller do not behave like that, so for
`DF_DRONE=f450` the simulation runs this controller at the physics rate (240 Hz) instead.

Control chain (the same shape as ArduPilot's):
    velocity setpoint -> (accel-limited shaping) -> velocity PI + feed-forward -> desired thrust vector
    -> tilt limit -> desired attitude -> attitude P -> body-rate P with gyro feed-forward -> torques
    -> motor mixer -> per-motor thrust -> RPM -> first-order motor lag -> PyBullet applies the forces.
"""
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel

G = 9.81
MOTOR_XY = 0.159                      # m, motor offset on each axis (0.225 m arm / sqrt 2); see f450.urdf
MOTOR_POS = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]]) * MOTOR_XY     # same order as the props in the URDF
YAW_SIGN = np.array([-1.0, 1.0, -1.0, 1.0])                               # BaseAviary's z-torque convention


class F450Model:
    """Stands in for a DroneModel member: loads `f450.urdf`, and behaves like the X-configuration
    Crazyflie wherever BaseAviary branches on the model (motor layout and torque signs are the same)."""
    value = "f450"
    name = "F450"

    def __eq__(self, other):
        return other is DroneModel.CF2X or isinstance(other, F450Model)

    def __hash__(self):
        return hash("f450")


F450 = F450Model()


class WindField:
    """Steady wind plus gusts (Ornstein-Uhlenbeck), in the world frame."""

    def __init__(self, rng: np.random.Generator, mean_speed: float = 0.0, gust: float = 0.0, tau: float = 2.0):
        self.rng, self.gust, self.tau = rng, gust, tau
        ang = rng.uniform(0, 2 * np.pi)
        self.mean = mean_speed * np.array([np.cos(ang), np.sin(ang), 0.0])
        self.dev = np.zeros(3)

    def step(self, dt: float) -> np.ndarray:
        if self.gust > 0:
            self.dev += -self.dev / self.tau * dt + self.gust * np.sqrt(2 * dt / self.tau) * self.rng.standard_normal(3) * [1, 1, .3]
        return self.mean + self.dev


class F450Controller:
    """Cascaded velocity -> attitude -> rate controller. State lives here; `reset` between episodes."""

    def __init__(self, mass, inertia_diag, kf, km, max_rpm, dt, tilt_max_deg=32.0, accel_max=4.0,
                 motor_tau=0.04, wind_cda=0.04):
        self.m, self.J, self.kf, self.dt = mass, np.diag(inertia_diag), kf, dt
        self.max_rpm, self.tilt_max, self.accel_max, self.motor_tau = max_rpm, np.deg2rad(tilt_max_deg), accel_max, motor_tau
        self.wind_cda = wind_cda
        self.f_max = kf * max_rpm ** 2
        c = km / kf
        self.mix = np.array([[1.0] * 4, MOTOR_POS[:, 1], -MOTOR_POS[:, 0], c * YAW_SIGN])      # [T, tx, ty, tz] = mix @ f
        self.mix_inv = np.linalg.inv(self.mix)
        self.kp_vel = np.array([2.6, 2.6, 4.0])
        self.ki_vel = np.array([0.6, 0.6, 1.2])
        self.kp_att = np.array([10.0, 10.0, 3.0])
        self.kp_rate = np.array([28.0, 28.0, 9.0])
        self.rate_max = np.array([5.0, 5.0, 2.0])
        self.reset()

    def reset(self, yaw: float = 0.0, rpm: float = 0.0):
        self.v_ref = None
        self.int = np.zeros(3)
        self.yaw = yaw
        self.rpm = np.full(4, rpm)

    def update(self, pos, quat, vel, ang_vel_world, v_target) -> np.ndarray:
        dt = self.dt
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        w_body = R.T @ ang_vel_world
        if self.v_ref is None:
            self.v_ref = vel.copy()
        step = np.clip(v_target - self.v_ref, -self.accel_max * dt, self.accel_max * dt)       # acceleration-limited shaping
        a_ff = step / dt
        self.v_ref = self.v_ref + step
        e_v = self.v_ref - vel
        self.int = np.clip(self.int + e_v * dt, -1.5, 1.5)
        a = self.kp_vel * e_v + self.ki_vel * self.int + a_ff
        a_xy = np.linalg.norm(a[:2])
        if a_xy > self.accel_max:
            a[:2] *= self.accel_max / a_xy
        a[2] = np.clip(a[2], -4.0, 5.0)
        f_des = self.m * (a + np.array([0.0, 0.0, G]))
        f_des[2] = max(f_des[2], 0.3 * self.m * G)
        lim = np.tan(self.tilt_max) * f_des[2]                               # tilt limit (ArduPilot ANGLE_MAX)
        h = np.linalg.norm(f_des[:2])
        if h > lim:
            f_des[:2] *= lim / h
        z_des = f_des / np.linalg.norm(f_des)
        thrust = float(np.clip(f_des @ R[:, 2], 0.0, 4 * self.f_max))
        xc = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        y_des = np.cross(z_des, xc)
        y_des /= np.linalg.norm(y_des)
        R_des = np.column_stack([np.cross(y_des, z_des), y_des, z_des])
        E = R_des.T @ R - R.T @ R_des
        e_R = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        w_des = np.clip(-self.kp_att * e_R, -self.rate_max, self.rate_max)
        tau = self.J @ (self.kp_rate * (w_des - w_body)) + np.cross(w_body, self.J @ w_body)
        f = np.clip(self.mix_inv @ np.array([thrust, *tau]), 0.02 * self.f_max, self.f_max)
        rpm_cmd = np.sqrt(f / self.kf)
        self.rpm += (rpm_cmd - self.rpm) * (1.0 - np.exp(-dt / self.motor_tau))             # motors do not respond instantly
        return self.rpm.copy()


class F450Mixin:
    """Mixed into FollowAvoidAviary. With DF_DRONE=cf2x every hook falls through to the library."""
    _f450 = False
    _fc: F450Controller | None = None
    _wind: WindField | None = None
    _v_target = np.zeros(3)

    def _f450_setup(self, wind: WindField | None):
        self._f450 = True
        self._wind = wind

    def _f450_set_wind(self, wind: WindField | None):
        self._wind = wind

    def _f450_reset(self):
        self._fc = None
        self._v_target = np.zeros(3)

    def _physics(self, rpm, nth_drone):
        if not self._f450:
            return super()._physics(rpm, nth_drone)
        if self._fc is None:                                                   # first physics step of an episode
            self._fc = F450Controller(self.M, np.array([self.J[0, 0], self.J[1, 1], self.J[2, 2]]), self.KF, self.KM,
                                      self.MAX_RPM, 1.0 / self.PYB_FREQ)
            quat0 = p.getBasePositionAndOrientation(self.DRONE_IDS[nth_drone], physicsClientId=self.CLIENT)[1]
            self._fc.reset(yaw=p.getEulerFromQuaternion(quat0)[2], rpm=self.HOVER_RPM)
        pos, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[nth_drone], physicsClientId=self.CLIENT)
        vel, wv = p.getBaseVelocity(self.DRONE_IDS[nth_drone], physicsClientId=self.CLIENT)
        vel = np.array(vel)
        cmd = self._fc.update(np.array(pos), quat, vel, np.array(wv), self._v_target)
        super()._physics(cmd, nth_drone)
        if self._wind is not None:                                             # quadratic drag against the moving air
            v_air = self._wind.step(1.0 / self.PYB_FREQ) - vel
            force = 0.5 * 1.2 * self._fc.wind_cda * np.linalg.norm(v_air) * v_air
            p.applyExternalForce(self.DRONE_IDS[nth_drone], -1, force.tolist(), list(pos), p.WORLD_FRAME,
                                 physicsClientId=self.CLIENT)
