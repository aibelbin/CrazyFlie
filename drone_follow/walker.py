"""A visual-only walking human figure (head, torso, swinging arms and legs) and scenery.

Nothing here has collision shapes: ray sensing and the collision check use the sphere
obstacles in `scenario.py`, so these only make the simulated world look like a place.
"""
import numpy as np
import pybullet as p

SKIN = (0.93, 0.76, 0.62, 1.0)
PANTS = (0.15, 0.2, 0.35, 1.0)

# (name, radius, cylinder length, hip/shoulder height, lateral offset, swing amplitude, phase sign)
_LIMBS = [
    ("leg_l", 0.075, 0.70, 0.85, +0.10, 0.50, +1.0),
    ("leg_r", 0.075, 0.70, 0.85, -0.10, 0.50, -1.0),
    ("arm_l", 0.05, 0.50, 1.40, +0.24, 0.40, -1.0),
    ("arm_r", 0.05, 0.50, 1.40, -0.24, 0.40, +1.0),
]


class Walker:
    def __init__(self, client: int, shirt_rgba: tuple):
        self.c = client
        self.phase = 0.0
        self.yaw = 0.0
        self.swing = 0.0
        self.torso = self._body(p.GEOM_CAPSULE, shirt_rgba, radius=0.17, length=0.26)
        self.head = self._body(p.GEOM_SPHERE, SKIN, radius=0.11)
        self.limbs = []
        for name, r, ln, top, lat, amp, sgn in _LIMBS:
            rgba = PANTS if name.startswith("leg") else shirt_rgba
            self.limbs.append((self._body(p.GEOM_CAPSULE, rgba, radius=r, length=ln),
                               ln + 2 * r, top, lat, amp, sgn))

    def _body(self, shape, rgba, **kw) -> int:
        vis = p.createVisualShape(shape, rgbaColor=rgba, physicsClientId=self.c, **kw)
        return p.createMultiBody(0, -1, vis, basePosition=[0, 0, -50], physicsClientId=self.c)

    def update(self, xy, vel_xy, dt: float):
        speed = float(np.linalg.norm(vel_xy))
        if speed > 0.05:
            self.yaw = float(np.arctan2(vel_xy[1], vel_xy[0]))
            self.phase += 2 * np.pi * (0.8 + 0.8 * speed) * dt
        target = min(1.0, speed / 0.5)
        self.swing += 0.2 * (target - self.swing)          # ease in/out of the gait
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        origin = np.array([xy[0], xy[1], 0.0])

        def world(local):
            return origin + np.array([cy * local[0] - sy * local[1],
                                      sy * local[0] + cy * local[1], local[2]])

        def place(body, local, pitch):
            quat = p.getQuaternionFromEuler([0, pitch, self.yaw])
            p.resetBasePositionAndOrientation(body, world(local).tolist(), quat, physicsClientId=self.c)

        place(self.torso, (0, 0, 1.15), 0.0)
        place(self.head, (0, 0, 1.58), 0.0)
        for body, length, top, lat, amp, sgn in self.limbs:
            theta = amp * self.swing * sgn * np.sin(self.phase)
            local = (-0.5 * length * np.sin(theta), lat, top - 0.5 * length * np.cos(theta))
            place(body, local, theta)


def build_scenery(client: int, rng: np.random.Generator):
    """Trees and buildings in a ring outside the flying arena, plus boundary posts."""
    def vis_body(shape, rgba, pos, **kw):
        v = p.createVisualShape(shape, rgbaColor=rgba, physicsClientId=client, **kw)
        p.createMultiBody(0, -1, v, basePosition=pos, physicsClientId=client)

    for k in range(14):                                          # trees
        ang = 2 * np.pi * (k + rng.uniform(-0.3, 0.3)) / 14
        r = rng.uniform(10.0, 13.0)
        x, y = r * np.cos(ang), r * np.sin(ang)
        h = rng.uniform(2.0, 3.0)
        vis_body(p.GEOM_CYLINDER, (0.4, 0.27, 0.15, 1), [x, y, h / 2], radius=0.18, length=h)
        vis_body(p.GEOM_SPHERE, (0.15, rng.uniform(0.45, 0.6), 0.2, 1), [x, y, h + 0.6],
                 radius=rng.uniform(0.9, 1.3))
    for k in range(5):                                           # buildings
        ang = 2 * np.pi * (k + 0.5) / 5
        x, y = 16 * np.cos(ang), 16 * np.sin(ang)
        w, hgt = rng.uniform(2.5, 4.0), rng.uniform(4.0, 8.0)
        g = rng.uniform(0.55, 0.8)
        vis_body(p.GEOM_BOX, (g, g, g + 0.05, 1), [x, y, hgt / 2], halfExtents=[w, w, hgt / 2])
    for x, y in [(-8, -8), (-8, 8), (8, -8), (8, 8)]:           # arena corner posts
        vis_body(p.GEOM_CYLINDER, (0.95, 0.75, 0.1, 1), [x, y, 1.0], radius=0.06, length=2.0)
