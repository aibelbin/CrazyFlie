"""One mission leg: start, waypoint, gentle terrain and static obstacles near the straight path.

The leg runs along +x from (0, 0) to (length, 0) at MISSION_ALT above the ground. Obstacles are vertical
cylinders (tree trunks, poles), spheres (tree canopies, rocks) and axis-aligned boxes (buildings, walls),
described in NumPy so the analytic clearance check and the PyBullet bodies the sensor sees agree.

Levels: 0 = no obstacles, 1 = sparse trees, 2 = dense trees, poles and rocks,
3 = level 2 plus buildings and a wall across the path (some are shorter to climb over than to go around).
"""
from dataclasses import dataclass, field

import numpy as np
import pybullet as p

from drone_follow import config as C

CELL = 1.0                  # m, heightfield resolution
MARGIN = 20.0               # m, terrain extends this far beyond the corridor
TRUNK_R, TRUNK_TOP = 0.25, 3.5
POLE_R = 0.15
SINK = 0.5                  # m, trunks, poles and boxes start this far underground so slopes leave no gap


class MissionTerrain:
    """Gentle hills along the corridor (open country: no mountains)."""

    def __init__(self, rng: np.random.Generator, length: float):
        n = max(4, int(length / 15))
        w = C.MISSION_HALF_WIDTH
        self.c = np.column_stack([rng.uniform(-10, length + 10, n), rng.uniform(-w, w, n)])
        self.sig = rng.uniform(8.0, 20.0, n)
        self.amp = rng.uniform(0.5, 3.0, n)
        self.x0, self.x1 = -MARGIN, length + MARGIN
        self.y0, self.y1 = -w - MARGIN, w + MARGIN

    def height(self, x, y) -> np.ndarray:
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = np.zeros(np.broadcast(x, y).shape)
        for c, s, a in zip(self.c, self.sig, self.amp):
            h = h + a * np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * s * s))
        return h

    def attach(self, client: int) -> int:
        xs = np.arange(self.x0, self.x1 + CELL / 2, CELL)
        ys = np.arange(self.y0, self.y1 + CELL / 2, CELL)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        z = self.height(gx, gy)                                       # [row = y, col = x]
        zmin, zmax = float(z.min()), float(z.max())
        # Bullet's heightfield: "rows" run along x (width), "columns" along y; data index = y * nx + x
        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD, meshScale=[CELL, CELL, 1.0], heightfieldData=z.flatten(order="C").tolist(),
            numHeightfieldRows=len(xs), numHeightfieldColumns=len(ys), physicsClientId=client)
        body = p.createMultiBody(0, shape, basePosition=[(xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2,
                                                         (zmin + zmax) / 2], physicsClientId=client)
        p.changeVisualShape(body, -1, rgbaColor=[0.35, 0.5, 0.25, 1], physicsClientId=client)
        return body


@dataclass
class Leg:
    length: float
    start: np.ndarray
    goal: np.ndarray
    terrain: MissionTerrain
    spheres: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))      # x, y, z, r
    cylinders: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))    # x, y, z0, z1, r
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))        # xmin, ymin, zmin, xmax, ymax, zmax
    sphere_kind: list = field(default_factory=list)
    cylinder_kind: list = field(default_factory=list)
    box_kind: list = field(default_factory=list)

    def ground(self, xy) -> float:
        return float(self.terrain.height(xy[0], xy[1]))

    def gaps(self, pos: np.ndarray):
        """Surface distance from `pos` to every obstacle (negative inside), and their kinds."""
        out = []
        s = self.spheres
        out.append(np.linalg.norm(s[:, :3] - pos, axis=1) - s[:, 3])
        c = self.cylinders
        radial = np.linalg.norm(c[:, :2] - pos[:2], axis=1) - c[:, 4]
        below, above = c[:, 2] - pos[2], pos[2] - c[:, 3]
        dz = np.maximum(np.maximum(below, above), 0.0)
        outside = np.hypot(np.maximum(radial, 0.0), dz)
        inside = np.maximum(radial, np.maximum(below, above))
        out.append(np.where((radial > 0) | (dz > 0), outside, inside))
        b = self.boxes
        d = np.maximum(b[:, :3] - pos, pos - b[:, 3:])
        out.append(np.where((d > 0).any(axis=1), np.linalg.norm(np.maximum(d, 0.0), axis=1), d.max(axis=1, initial=-np.inf)))
        return np.concatenate(out), self.sphere_kind + self.cylinder_kind + self.box_kind

    def clearance(self, pos: np.ndarray) -> float:
        g, _ = self.gaps(np.asarray(pos, float))
        return float(g.min()) - C.MISSION_DRONE_RADIUS if len(g) else np.inf

    def nearest_kind(self, pos: np.ndarray):
        g, kinds = self.gaps(np.asarray(pos, float))
        return kinds[int(np.argmin(g))] if len(g) else None

    def add_box(self, lo, hi, kind: str, client: int | None = None):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.boxes = np.vstack([self.boxes, np.concatenate([lo, hi])[None, :]])
        self.box_kind.append(kind)
        if client is not None:
            _make_box(client, lo, hi)


def make_leg(rng: np.random.Generator, level: int, length: float | None = None) -> Leg:
    L = float(length if length is not None else rng.uniform(*C.MISSION_LEG))
    terrain = MissionTerrain(rng, L)
    gz = lambda x, y: float(terrain.height(x, y))
    leg = Leg(L, np.array([0.0, 0.0, gz(0, 0) + C.MISSION_ALT]), np.array([L, 0.0, gz(L, 0) + C.MISSION_ALT]), terrain)
    spheres, cyls, boxes = [], [], []

    def spot(on_path: bool, spread: float):
        """A point along the leg, away from the start (8 m) and the waypoint (6 m)."""
        x = rng.uniform(15.0, L - 10.0)
        y = rng.uniform(-2.0, 2.0) if on_path else rng.uniform(-spread, spread)
        return x, y

    def trees(n, spread, on_path):
        for i in range(n):
            x, y = spot(i < on_path, spread)
            g, r = gz(x, y), rng.uniform(1.2, 2.0)
            cyls.append(([x, y, g - SINK, g + TRUNK_TOP, TRUNK_R], "tree"))
            spheres.append(([x, y, g + TRUNK_TOP + 0.6 * r, r], "tree"))

    def poles(n, spread, on_path):
        for i in range(n):
            x, y = spot(i < on_path, spread)
            g = gz(x, y)
            cyls.append(([x, y, g - SINK, g + rng.uniform(7.0, 12.0), POLE_R], "pole"))

    def rocks(n, spread):
        for _ in range(n):
            x, y = spot(False, spread)
            r = rng.uniform(0.5, 1.5)
            spheres.append(([x, y, gz(x, y) + 0.4 * r, r], "rock"))

    def box(x0, x1, y0, y1, height, kind):
        g = min(gz(x, y) for x in (x0, x1) for y in (y0, y1))
        boxes.append(([x0, y0, g - SINK, x1, y1, gz((x0 + x1) / 2, (y0 + y1) / 2) + height], kind))

    if level >= 1:
        trees(max(2, int(L / 15)), 12.0, on_path=2)
    if level >= 2:
        trees(int(L / 6), 10.0, on_path=4)
        poles(max(2, int(L / 20)), 8.0, on_path=2)
        rocks(int(L / 15), 10.0)
    if level >= 3:
        for _ in range(int(rng.integers(1, 3))):                         # buildings on the path
            w, d = rng.uniform(6.0, 15.0, 2)
            lo = 25.0 + w / 2                                            # short test legs: keep the range valid
            cx, cy = rng.uniform(lo, max(lo, L - 25.0 - w / 2)), rng.uniform(-4.0, 4.0)
            box(cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2, rng.uniform(6.0, 15.0), "building")
        wx, span, wy = rng.uniform(30.0, L - 30.0), rng.uniform(14.0, 30.0), rng.uniform(-5.0, 5.0)
        box(wx - 0.3, wx + 0.3, wy - span / 2, wy + span / 2, rng.uniform(6.0, 9.0), "wall")   # across the path

    if spheres:
        leg.spheres, leg.sphere_kind = np.array([s for s, _ in spheres]), [k for _, k in spheres]
    if cyls:
        leg.cylinders, leg.cylinder_kind = np.array([c for c, _ in cyls]), [k for _, k in cyls]
    if boxes:
        leg.boxes, leg.box_kind = np.array([b for b, _ in boxes]), [k for _, k in boxes]
    return leg


def _make_box(client: int, lo: np.ndarray, hi: np.ndarray):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=((hi - lo) / 2).tolist(), physicsClientId=client)
    p.createMultiBody(0, col, basePosition=((lo + hi) / 2).tolist(), physicsClientId=client)


def attach_leg(leg: Leg, client: int):
    """PyBullet bodies for the terrain and every obstacle, so the range sensor sees them."""
    leg.terrain.attach(client)
    for x, y, z, r in leg.spheres:
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=float(r), physicsClientId=client)
        p.createMultiBody(0, col, basePosition=[x, y, z], physicsClientId=client)
    for x, y, z0, z1, r in leg.cylinders:
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(r), height=float(z1 - z0), physicsClientId=client)
        p.createMultiBody(0, col, basePosition=[x, y, (z0 + z1) / 2], physicsClientId=client)
    for row in leg.boxes:
        _make_box(client, row[:3], row[3:])
