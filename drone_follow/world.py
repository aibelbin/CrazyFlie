"""Outdoor world for difficulty >= 4: rolling hills with mountains at the edge, a forest and rocks.

Everything is deterministic given the scenario's random generator. Trees and rocks are made of spheres
so the ray-cast sensor and the analytic collision check treat them like any other obstacle.
"""
import numpy as np
import pybullet as p

HALF = 30.0                       # the terrain covers +-HALF metres
GRID = 121                        # heightfield samples per side (0.5 m cells)
TREE_SPHERES = ((0.45, 0.28), (1.2, 0.26), (1.95, 0.24), (3.2, 1.1))    # (height above ground, radius): trunk x3, canopy


class Terrain:
    """Height of the ground at any (x, y): gentle hills inside, tall mountains toward the edge."""

    def __init__(self, rng: np.random.Generator):
        n_hills = 14
        self.c = rng.uniform(-18, 18, (n_hills, 2))
        self.sig = rng.uniform(3.0, 7.0, n_hills)
        self.amp = rng.uniform(0.5, 2.4, n_hills)
        ang = rng.uniform(0, 2 * np.pi, 7)
        self.mc = 33.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)       # mountains outside the flying area
        self.msig = rng.uniform(9.0, 14.0, 7)
        self.mamp = rng.uniform(9.0, 16.0, 7)

    def height(self, x, y) -> np.ndarray:
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = np.zeros(np.broadcast(x, y).shape)
        for c, s, a in zip(self.c, self.sig, self.amp):
            h = h + a * np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * s * s))
        for c, s, a in zip(self.mc, self.msig, self.mamp):
            h = h + a * np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * s * s))
        return h

    def grid(self) -> np.ndarray:
        xs = np.linspace(-HALF, HALF, GRID)
        gx, gy = np.meshgrid(xs, xs, indexing="xy")
        return self.height(gx, gy)                           # [row = y, col = x]

    def attach(self, client: int) -> int:
        """A PyBullet heightfield body, so range rays hit the ground. Returns the body id."""
        z = self.grid()
        zmin, zmax = float(z.min()), float(z.max())
        cell = 2 * HALF / (GRID - 1)
        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD, meshScale=[cell, cell, 1.0], heightfieldData=z.flatten(order="C").tolist(),
            numHeightfieldRows=GRID, numHeightfieldColumns=GRID, physicsClientId=client)
        body = p.createMultiBody(0, shape, basePosition=[0, 0, (zmin + zmax) / 2], physicsClientId=client)
        p.changeVisualShape(body, -1, rgbaColor=[0.35, 0.5, 0.25, 1], physicsClientId=client)
        return body


def forest(rng: np.random.Generator, terrain: Terrain, keep_clear: list, n_trees: int = 34, n_rocks: int = 14,
           half: float = 20.0):
    """Static obstacles: returns (centres Nx3, radii N, kinds N, group N). `keep_clear` = [(xy, radius)]."""
    pos, rad, kind, group = [], [], [], []

    def free(xy, margin):
        return all(np.linalg.norm(xy - np.asarray(c)) > r + margin for c, r in keep_clear)

    g = 0
    for _ in range(n_trees):
        for _try in range(30):
            xy = rng.uniform(-half, half, 2)
            if free(xy, 0.5):
                break
        else:
            continue
        gz = float(terrain.height(xy[0], xy[1]))
        for dz, r in TREE_SPHERES:
            pos.append([xy[0], xy[1], gz + dz]); rad.append(r); kind.append("tree"); group.append(g)
        g += 1
    for _ in range(n_rocks):
        for _try in range(30):
            xy = rng.uniform(-half, half, 2)
            if free(xy, 0.5):
                break
        else:
            continue
        r = rng.uniform(0.35, 0.9)
        pos.append([xy[0], xy[1], float(terrain.height(xy[0], xy[1])) + 0.45 * r])
        rad.append(r); kind.append("rock"); group.append(g)
        g += 1
    return (np.array(pos).reshape(-1, 3), np.array(rad), np.array(kind, dtype=object), np.array(group, dtype=int))
