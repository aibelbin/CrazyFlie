"""A visible quadcopter for the live demo: the Crazyflie mesh scaled up, with spinning blades.

The physics body stays 9 cm and untouched (and is hidden); this follows its pose, so what
you see is exactly what the simulation flies.
"""
from importlib.resources import files

import numpy as np
import pybullet as p

SCALE = 5.0                     # 9 cm Crazyflie -> ~45 cm, readable from several metres
MOTORS = [(0.028, 0.028), (-0.028, 0.028), (-0.028, -0.028), (0.028, -0.028)]
BLADE_Z = 0.018
SPIN_RAD_PER_STEP = 2.2         # visual spin per rendered update (spin direction alternates)


class DroneVisual:
    def __init__(self, client: int, drone_id: int):
        self.c, self.drone_id, self.spin = client, drone_id, 0.0
        for link in range(-1, p.getNumJoints(drone_id, physicsClientId=client)):
            p.changeVisualShape(drone_id, link, rgbaColor=[0, 0, 0, 0], physicsClientId=client)
        mesh = str(files("gym_pybullet_drones") / "assets" / "cf2.dae")
        vis = p.createVisualShape(
            p.GEOM_MESH, fileName=mesh, meshScale=[SCALE] * 3, rgbaColor=[0.25, 0.25, 0.28, 1],
            visualFrameOrientation=p.getQuaternionFromEuler([0, 0, 55]), physicsClientId=client)
        self.body = p.createMultiBody(0, -1, vis, basePosition=[0, 0, -50], physicsClientId=client)
        blade = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.032 * SCALE, 0.0035 * SCALE, 0.0008 * SCALE],
            rgbaColor=[0.9, 0.9, 0.95, 0.9], physicsClientId=client)
        self.blades = [p.createMultiBody(0, -1, blade, basePosition=[0, 0, -50], physicsClientId=client)
                       for _ in MOTORS]

    def update(self):
        c = self.c
        pos, quat = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=c)
        p.resetBasePositionAndOrientation(self.body, pos, quat, physicsClientId=c)
        self.spin += SPIN_RAD_PER_STEP
        for i, (body, (mx, my)) in enumerate(zip(self.blades, MOTORS)):
            sign = 1.0 if i % 2 == 0 else -1.0
            local_pos = [mx * SCALE, my * SCALE, BLADE_Z * SCALE]
            spin_q = p.getQuaternionFromEuler([0, 0, sign * self.spin + i * 0.8])
            wpos, wquat = p.multiplyTransforms(pos, quat, local_pos, spin_q)
            p.resetBasePositionAndOrientation(body, wpos, wquat, physicsClientId=c)
