# Waypoint Mission Avoidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A simulated waypoint-mission task for the F450 where classical avoiders (potential field, BendyRuler-style) and a PPO avoider are scored on the same seeded legs under an ArduPilot-style AUTO/GUIDED supervisor.

**Architecture:** New, separate modules beside the follow-the-person code: `mission_world.py` (leg, terrain, obstacles, analytic clearance), `mission_env.py` (`MissionAviary`, track-frame observations, rays cast in the track frame), `mission_policies.py` (straight line, potential field, BendyRuler, `Supervisor`), `mission_evaluate.py` (seeded scoring + table), `mission_train.py` (PPO, reusing `train_long`'s callbacks). The follow env is untouched except a backward-compatible generalisation of `sensing.py`.

**Tech Stack:** Python 3.11, PyBullet, gym-pybullet-drones (vendored), Gymnasium, Stable-Baselines3 PPO, NumPy, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-27-waypoint-mission-avoidance-design.md`.
- **Never train or run long jobs on the Mac** (fanless MacBook Air). On the Mac: unit tests and tiny smoke tests only. All training and the 500-leg evaluations run on `gpu-sys` (WSL Ubuntu, user `aibel_win`, repo `~/DetectionAndAvoidance`, env via `~/bin/micromamba run -n df`).
- **Ask the user before copying code to gpu-sys and before launching each job there.**
- gpu-sys traps: send multi-line scripts over stdin (`ssh gpu-sys 'wsl -e bash -s' < script.sh`), never put `|`, `^` or `$` inline in the ssh command, start long jobs detached with WMI (command in Task 7), poll with short ssh calls.
- Local tests: `.venv/bin/python -m pytest ...` from the repo root.
- Values from the spec: leg 80-150 m, corridor about 200 m x 60 m, cruise 5 m/s, speed limit 6 m/s, sensor range 20 m with up/down rays, success within 2 m, timeout 3x straight-line time, handover trigger about 12 m, 500 legs per level, RL 5M steps x 2 seeds, levels 0-3 (0 none, 1 sparse trees, 2 dense trees and poles, 3 adds buildings and walls).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `drone_follow/config.py` (modify) | Mission constants (new section at the end) |
| `drone_follow/sensing.py` (modify) | `ray_directions()`, `cast()`; `range_scan` unchanged in behaviour |
| `drone_follow/mission_world.py` (create) | `MissionTerrain`, `Leg` (clearance, kinds, `add_box`), `make_leg`, `attach_leg` |
| `drone_follow/mission_env.py` (create) | `MissionAviary`, `MissionWrapper`, `make_mission_env`, obs layout constants |
| `drone_follow/mission_policies.py` (create) | `straight_line`, `potential_field`, `BendyRuler`, `Supervisor`, `load_ppo` |
| `drone_follow/mission_evaluate.py` (create) | `run_legs`, `make_policy`, `markdown_table`, CLI |
| `drone_follow/mission_train.py` (create) | `MissionEval` callback, PPO training CLI |
| `tests/test_mission_world.py`, `tests/test_mission_env.py`, `tests/test_mission_policies.py`, `tests/test_mission_evaluate.py`, `tests/test_mission_train.py` (create) | tests |
| `README.md` (modify) | results table (Task 7) |

---

### Task 1: Mission constants and generalised ray casting

**Files:**
- Modify: `drone_follow/config.py` (append)
- Modify: `drone_follow/sensing.py` (whole file, 27 lines)
- Test: `tests/test_env.py` (append one test)

**Interfaces:**
- Produces: `config.MISSION_*` / `config.MR_*` constants below; `sensing.ray_directions(n_az: int, elevs_deg) -> np.ndarray (N, 3)` (elevation-major: index = e * n_az + k, k = 0 points along +x); `sensing.cast(client: int, pos: np.ndarray, dirs: np.ndarray, max_range: float, start_offset: float = C.RAY_START_OFFSET) -> np.ndarray (N,)`. `RAY_DIRS`, `N_RAYS`, `range_scan` keep their current meaning.

- [ ] **Step 1: Append the mission constants to `drone_follow/config.py`**

```python

# Mission legs (waypoint flight with obstacle avoidance; always the F450 with sensor noise, see mission_env.py)
MISSION_CRUISE = 5.0            # m/s, ArduPilot's default WPNAV_SPEED
MISSION_SPEED_LIMIT = 6.0       # m/s at |action| = 1
MISSION_ALT = 5.0               # m above the ground at the start and at the waypoint
MISSION_CEILING = 30.0          # m above the ground; higher ends the leg as a failure
MISSION_LEG = (80.0, 150.0)     # m, range of leg lengths
MISSION_HALF_WIDTH = 30.0       # m, corridor half-width; leaving it ends the leg as a failure
MISSION_GOAL_RADIUS = 2.0       # m, success when this close to the waypoint
MISSION_TIME_FACTOR = 3.0       # timeout = this x the straight-line time at cruise speed
MISSION_RANGE = 20.0            # m, sensor range (stopping from 5 m/s alone takes ~4-5 m)
MISSION_N_AZ = 16
MISSION_ELEVS_DEG = (-35.0, -15.0, 0.0, 15.0, 35.0)
MISSION_RAY_START = 0.4         # m, rays start outside the F450's 0.23 m collision cylinder
MISSION_DRONE_RADIUS = 0.32     # m, F450 frame + props (config.DRONE_RADIUS depends on DF_DRONE)
MISSION_ACT_DELAY = 1           # control steps between a command and the autopilot using it
# Mission reward
MR_PROGRESS = 0.1               # per metre gained toward the waypoint
MR_SUCCESS = 10.0
MR_CRASH = -20.0
MR_FAIL = -10.0                 # timeout, leaving the corridor or climbing above the ceiling
MR_PROXIMITY = 0.05             # per step at worst, for level or upward rays closer than MR_PROXIMITY_START
MR_PROXIMITY_START = 3.0        # m
MR_SMOOTH = 0.02                # per unit of command change
```

- [ ] **Step 2: Write the failing test** (append to `tests/test_env.py`)

```python
def test_cast_takes_any_directions_and_range(client):
    from drone_follow.sensing import cast, ray_directions
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5, physicsClientId=client)
    p.createMultiBody(0, col, basePosition=[15, 0, 1], physicsClientId=client)
    dirs = ray_directions(16, (0.0,))
    assert dirs.shape == (16, 3) and np.allclose(dirs[0], [1, 0, 0]) and np.allclose(dirs[4], [0, 1, 0], atol=1e-9)
    r = cast(client, np.array([0.0, 0.0, 1.0]), dirs, 20.0, start_offset=0.4)
    assert r[0] == pytest.approx(14.5, abs=0.05)              # sphere surface at x = 14.5
    assert r[8] == pytest.approx(20.0)                        # nothing behind
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_env.py::test_cast_takes_any_directions_and_range -v`
Expected: FAIL with `ImportError: cannot import name 'cast'`

- [ ] **Step 4: Replace `drone_follow/sensing.py` with**

```python
"""Range sensing: a fan of ray casts around the drone, via PyBullet's ray tester."""
import numpy as np
import pybullet as p

from drone_follow import config as C


def ray_directions(n_az: int, elevs_deg) -> np.ndarray:
    """Unit vectors, elevation-major: index = e * n_az + k, with k = 0 along +x and k increasing toward +y."""
    dirs = []
    for el in np.radians(elevs_deg):
        for k in range(n_az):
            az = 2 * np.pi * k / n_az
            dirs.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.array(dirs)


RAY_DIRS = ray_directions(C.N_AZ, C.ELEVS_DEG)
N_RAYS = len(RAY_DIRS)


def cast(client: int, pos: np.ndarray, dirs: np.ndarray, max_range: float,
         start_offset: float = C.RAY_START_OFFSET) -> np.ndarray:
    """Distance to the nearest surface along each direction, capped at `max_range`."""
    starts = pos + dirs * start_offset
    ends = pos + dirs * max_range
    hits = p.rayTestBatch(starts.tolist(), ends.tolist(), physicsClientId=client)
    frac = np.fromiter((h[2] for h in hits), dtype=float, count=len(dirs))
    return start_offset + frac * (max_range - start_offset)


def range_scan(client: int, pos: np.ndarray) -> np.ndarray:
    """The follow env's strip: distance to the nearest surface along each ray, capped at MAX_RANGE."""
    return cast(client, pos, RAY_DIRS, C.MAX_RANGE)
```

- [ ] **Step 5: Run the sensing and env tests**

Run: `.venv/bin/python -m pytest tests/test_env.py -q`
Expected: all pass (including the existing `test_range_scan_sees_a_sphere`).

- [ ] **Step 6: Commit**

```bash
git add drone_follow/config.py drone_follow/sensing.py tests/test_env.py
git commit -m "Mission constants; ray casting for any directions and range

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Mission world (leg, terrain, obstacles, clearance)

**Files:**
- Create: `drone_follow/mission_world.py`
- Test: `tests/test_mission_world.py`

**Interfaces:**
- Consumes: `config.MISSION_*` from Task 1.
- Produces:
  - `MissionTerrain(rng, length)` with `.height(x, y) -> np.ndarray` and `.attach(client) -> int`.
  - `Leg` dataclass: `length: float`, `start: np.ndarray (3,)`, `goal: np.ndarray (3,)`, `terrain: MissionTerrain`, `spheres: (N, 4)` rows `x, y, z, r`, `cylinders: (M, 5)` rows `x, y, z0, z1, r`, `boxes: (K, 6)` rows `xmin, ymin, zmin, xmax, ymax, zmax`, `sphere_kind / cylinder_kind / box_kind: list[str]`. Methods: `ground(xy) -> float`, `gaps(pos) -> (np.ndarray, list[str])` (surface distance per obstacle, negative inside), `clearance(pos) -> float` (min gap minus `MISSION_DRONE_RADIUS`, `inf` if no obstacles), `nearest_kind(pos) -> str | None`, `add_box(lo, hi, kind, client=None)`.
  - `make_leg(rng, level: int, length: float | None = None) -> Leg` (leg along +x from (0, 0) to (length, 0)).
  - `attach_leg(leg, client) -> None` creates the terrain and every obstacle in PyBullet.
  - Kinds used: `"tree"` (trunk cylinder + canopy sphere), `"pole"`, `"rock"`, `"building"`, `"wall"`.

- [ ] **Step 1: Write the failing tests** (`tests/test_mission_world.py`)

```python
import numpy as np
import pybullet as p
import pytest

from drone_follow import config as C
from drone_follow.mission_world import MissionTerrain, attach_leg, make_leg


def test_heightfield_matches_the_height_function_on_a_long_corridor():
    client = p.connect(p.DIRECT)
    try:
        t = MissionTerrain(np.random.default_rng(5), 120.0)
        t.attach(client)
        rng = np.random.default_rng(0)
        for _ in range(25):
            x, y = rng.uniform(-10, 130), rng.uniform(-28, 28)
            hit = p.rayTest([x, y, 60], [x, y, -5], physicsClientId=client)[0]
            assert abs((60 - 65 * hit[2]) - float(t.height(x, y))) < 0.05
    finally:
        p.disconnect(client)


def test_leg_runs_along_x_at_mission_altitude():
    for seed in range(10):
        leg = make_leg(np.random.default_rng(seed), 2)
        assert C.MISSION_LEG[0] <= leg.length <= C.MISSION_LEG[1]
        assert np.allclose(leg.start[:2], [0, 0]) and np.allclose(leg.goal[:2], [leg.length, 0])
        for pt in (leg.start, leg.goal):
            assert pt[2] - leg.ground(pt[:2]) == pytest.approx(C.MISSION_ALT)
            assert leg.clearance(pt) > 3.0                        # nothing at the start or the waypoint


def test_levels_add_the_right_obstacles():
    def kinds(level):
        leg = make_leg(np.random.default_rng(3), level, length=120.0)
        return set(leg.sphere_kind) | set(leg.cylinder_kind) | set(leg.box_kind)
    assert kinds(0) == set()
    assert kinds(1) == {"tree"}
    assert {"tree", "pole", "rock"} <= kinds(2) and "wall" not in kinds(2)
    assert {"building", "wall"} <= kinds(3)


def test_some_obstacles_stand_on_the_straight_path():
    for seed in range(5):
        for level in (1, 2, 3):
            leg = make_leg(np.random.default_rng(seed), level, length=120.0)
            on_path = (np.abs(leg.cylinders[:, 1]) < 2.0).sum() if len(leg.cylinders) else 0
            assert on_path >= 2
        boxes = make_leg(np.random.default_rng(seed), 3, length=120.0).boxes
        assert ((boxes[:, 1] < 0) & (boxes[:, 4] > 0)).any()      # a building or the wall spans y = 0


def test_clearance_of_each_shape():
    leg = make_leg(np.random.default_rng(0), 0, length=100.0)
    g = leg.ground([50.0, 0.0])
    leg.add_box([50, -5, g], [51, 5, g + 8], "wall")
    pos = np.array([47.0, 0.0, g + 4])
    assert leg.clearance(pos) == pytest.approx(3.0 - C.MISSION_DRONE_RADIUS)
    assert leg.nearest_kind(pos) == "wall"
    assert leg.clearance(np.array([50.5, 0.0, g + 4])) < 0        # inside
    over = np.array([50.5, 0.0, g + 10])
    assert leg.clearance(over) == pytest.approx(2.0 - C.MISSION_DRONE_RADIUS)


def test_every_obstacle_exists_in_pybullet_up_to_its_top():
    from drone_follow.sensing import cast
    client = p.connect(p.DIRECT)
    try:
        leg = make_leg(np.random.default_rng(4), 3, length=100.0)
        attach_leg(leg, client)
        down = np.array([[0, 0, -1.0]])
        tops = ([(s[0], s[1], s[2] + s[3]) for s in leg.spheres] + [(c[0], c[1], c[3]) for c in leg.cylinders]
                + [((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, b[5]) for b in leg.boxes])
        for x, y, top in tops:                       # a ray dropped on its centre stops at its top, or on something taller
            hit_z = 60.0 - cast(client, np.array([x, y, 60.0]), down, 80.0, 0.0)[0]
            assert hit_z >= top - 0.05
    finally:
        p.disconnect(client)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_mission_world.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'drone_follow.mission_world'`

- [ ] **Step 3: Create `drone_follow/mission_world.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_mission_world.py -q`
Expected: 6 passed. If only `test_heightfield_matches_the_height_function_on_a_long_corridor` fails with errors of several metres, the heightfield axes are transposed: swap `numHeightfieldRows` and `numHeightfieldColumns` in `MissionTerrain.attach` and rerun.

- [ ] **Step 5: Commit**

```bash
git add drone_follow/mission_world.py tests/test_mission_world.py
git commit -m "Mission world: leg along +x, gentle terrain, trees/poles/rocks/buildings/walls, analytic clearance

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `MissionAviary` environment

**Files:**
- Create: `drone_follow/mission_env.py`
- Test: `tests/test_mission_env.py`

**Interfaces:**
- Consumes: `make_leg`, `attach_leg`, `Leg.clearance/ground/nearest_kind/add_box` (Task 2); `cast`, `ray_directions` (Task 1); `F450`, `F450Mixin`, `WindField` from `drone_follow/f450.py` (existing).
- Produces:
  - Constants: `M_RAY_DIRS (N, 3)` (track frame), `M_N_RAYS = 80`, `M_OBS_DIM = 11 + 2 * M_N_RAYS = 171`, `M_RANGES = slice(11, 91)`, `M_PREV_ACTION = slice(8, 11)`.
  - Observation layout: `0` horizontal distance to the waypoint / 100 (capped at 1.5); `1` waypoint height above the drone / 10; `2:5` velocity (track frame) / `MISSION_SPEED_LIMIT`; `5:7` roll, pitch; `7` height above ground / 10; `8:11` previous action; `M_RANGES` ranges / `MISSION_RANGE`; `91:171` range rate (closing speed / 5 m/s) clipped to ±1.5.
  - Action: track-frame velocity in `[-1, 1]^3`, 1 = `MISSION_SPEED_LIMIT`; norm clipped to 1.
  - `MissionAviary(level=1, seed=None, gui=False, length=None)`, `.set_level(level)`, `.leg`, `.pos`, `.vel`.
  - Info keys: `collision: bool`, `collision_kind: str | None` (`tree/pole/rock/building/wall/ground/flip`), `success: bool`, `failure: str | None` (`"timeout"` or `"out_of_bounds"`), `dist_to_goal`, `path_m`, `leg_m` (3D start-to-goal distance), `time_s`, `min_range`.
  - `MissionWrapper(env)` flattens obs to `(M_OBS_DIM,)` and actions to `(3,)`, exposes `set_level`.
  - `make_mission_env(level=1, seed=None, length=None, gui=False) -> thunk`.

- [ ] **Step 1: Write the failing tests** (`tests/test_mission_env.py`)

```python
import numpy as np
import pybullet as p
import pytest

from drone_follow import config as C
from drone_follow.mission_env import M_N_RAYS, M_OBS_DIM, M_RANGES, make_mission_env

FWD = 2 * C.MISSION_N_AZ                       # elevation 0, azimuth 0: straight at the waypoint


def straight(obs):
    g = np.array([obs[0] * 100.0, 0.0, obs[1] * 10.0])
    return C.MISSION_CRUISE / C.MISSION_SPEED_LIMIT * g / np.linalg.norm(g)


def test_shapes_and_finite_observations():
    env = make_mission_env(level=3, seed=0, length=60.0)()
    obs, info = env.reset(seed=0)
    assert obs.shape == (M_OBS_DIM,) and M_N_RAYS == 80
    for _ in range(30):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
        if term or trunc:
            break
    env.close()


def test_rays_do_not_see_the_drone_itself():
    env = make_mission_env(level=0, seed=1, length=40.0)()
    obs, _ = env.reset(seed=1)
    for _ in range(10):
        obs, *_ = env.step(np.zeros(3))
    level_and_up = obs[M_RANGES][2 * C.MISSION_N_AZ:] * C.MISSION_RANGE
    assert level_and_up.min() > 1.0                   # a ray hitting the drone's own body would read ~0.4 m
    env.close()


def test_straight_line_reaches_the_waypoint_on_an_empty_leg():
    env = make_mission_env(level=0, seed=2, length=25.0)()
    obs, _ = env.reset(seed=2)
    for _ in range(600):
        obs, r, term, trunc, info = env.step(straight(obs))
        if term or trunc:
            break
    assert info["success"] and not info["collision"] and info["failure"] is None
    assert info["path_m"] == pytest.approx(info["leg_m"], rel=0.1)
    assert info["time_s"] < 3 * 25.0 / C.MISSION_CRUISE
    env.close()


def test_forward_ray_sees_a_wall_in_the_track_frame_and_flying_into_it_crashes():
    env = make_mission_env(level=0, seed=3, length=60.0)()
    obs, _ = env.reset(seed=3)
    sim = env.unwrapped
    g = sim.leg.ground([12.0, 0.0])
    sim.leg.add_box([12, -6, g - 0.5], [13, 6, g + 15], "wall", client=sim.CLIENT)
    for _ in range(6):                                # let the delayed, smoothed sensor settle
        obs, *_ = env.step(np.zeros(3))
    assert obs[M_RANGES][FWD] * C.MISSION_RANGE == pytest.approx(12.0 - sim.pos[0][0], abs=0.6)
    for _ in range(300):
        obs, r, term, trunc, info = env.step(straight(obs))
        if term or trunc:
            break
    assert term and info["collision"] and info["collision_kind"] == "wall" and r == C.MR_CRASH
    env.close()


def test_same_seed_gives_same_leg_and_rewards():
    def rollout():
        env = make_mission_env(level=2, seed=5, length=50.0)()
        obs, _ = env.reset(seed=5)
        out = []
        for _ in range(20):
            obs, r, *_ = env.step(straight(obs))
            out.append(r)
        env.close()
        return out
    assert rollout() == pytest.approx(rollout())


def test_leaving_the_corridor_is_a_failure():
    env = make_mission_env(level=0, seed=6, length=80.0)()
    env.reset(seed=6)
    sim = env.unwrapped
    p.resetBasePositionAndOrientation(sim.DRONE_IDS[0], [10.0, C.MISSION_HALF_WIDTH + 1.0, sim.leg.start[2]],
                                      [0, 0, 0, 1], physicsClientId=sim.CLIENT)
    obs, r, term, trunc, info = env.step(np.zeros(3))
    assert trunc and not term and info["failure"] == "out_of_bounds" and r < 0
    env.close()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_mission_env.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'drone_follow.mission_env'`

- [ ] **Step 3: Create `drone_follow/mission_env.py`**

```python
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
from drone_follow.sensing import cast, ray_directions

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
        self._range_hist = deque(maxlen=C.SENSOR_DELAY + 1)
        self._range_smooth = None
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
    def _sense(self, true_ranges: np.ndarray) -> np.ndarray:
        """Depth sensor: delayed, noisy, occasional dropouts, lightly smoothed (the same model as env.py)."""
        self._range_hist.append(true_ranges)
        r = self._range_hist[0].copy()
        rel_sigma, abs_sigma = C.RANGE_NOISE
        r = r + self._rng.standard_normal(r.shape) * (rel_sigma * r + abs_sigma)
        r[self._rng.random(r.shape) < C.RANGE_DROPOUT] = C.MISSION_RANGE
        r = np.clip(r, 0.05, C.MISSION_RANGE)
        self._range_smooth = r if self._range_smooth is None else 0.5 * self._range_smooth + 0.5 * r
        return self._range_smooth.copy()

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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_mission_env.py -q`
Expected: 6 passed (a few seconds each; the longest flies ~200 control steps).

- [ ] **Step 5: Run the whole suite** (the follow env must be unaffected)

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add drone_follow/mission_env.py tests/test_mission_env.py
git commit -m "MissionAviary: F450 flies one leg to a waypoint, track-frame observations and rays, success/crash/failure outcomes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Avoiders and the AUTO/GUIDED supervisor

**Files:**
- Create: `drone_follow/mission_policies.py`
- Test: `tests/test_mission_policies.py`

**Interfaces:**
- Consumes: `M_OBS_DIM`, `M_RANGES`, `M_RAY_DIRS` (Task 3).
- Produces (all take one observation `np.ndarray (M_OBS_DIM,)` and return a track-frame action `np.ndarray (3,)`):
  - `CRUISE = MISSION_CRUISE / MISSION_SPEED_LIMIT`.
  - `straight_line(obs)`.
  - `potential_field(obs, influence=6.0, gain=1.5)`.
  - `BendyRuler(lookahead=10.0, hysteresis_deg=10.0, min_elev=-15.0)`: callable with `.reset()`.
  - `Supervisor(avoider, trigger=12.0, margin=3.0, hold_s=0.5, cone_deg=45.0)`: callable with `.reset()` and attribute `.guided: bool`.
  - `load_ppo(path) -> callable`.

- [ ] **Step 1: Write the failing tests** (`tests/test_mission_policies.py`)

```python
import numpy as np
import pytest

from drone_follow import config as C
from drone_follow.mission_env import M_OBS_DIM, M_RANGES
from drone_follow.mission_policies import CRUISE, BendyRuler, Supervisor, potential_field, straight_line

ELEVS = np.array(C.MISSION_ELEVS_DEG)
N_AZ = C.MISSION_N_AZ
AZ = (np.arange(N_AZ) * 360.0 / N_AZ + 180) % 360 - 180          # 0 = toward the waypoint


def make_obs(dist=50.0, blocked=()):
    """`blocked` = [(max |azimuth| deg, set of elevations, range m)] applied to an all-clear strip."""
    obs = np.zeros(M_OBS_DIM, dtype=np.float32)
    obs[0] = dist / 100.0
    r = np.full((len(ELEVS), N_AZ), C.MISSION_RANGE)
    for az_max, elevs, rng in blocked:
        for e, el in enumerate(ELEVS):
            if el in elevs:
                r[e, np.abs(AZ) <= az_max] = rng
    obs[M_RANGES] = r.flatten() / C.MISSION_RANGE
    return obs


def test_straight_line_flies_at_the_waypoint_at_cruise_speed():
    assert straight_line(make_obs()) == pytest.approx([CRUISE, 0.0, 0.0])


def test_potential_field_pushes_away_from_an_obstacle_on_the_left():
    obs = make_obs()
    r = obs[M_RANGES].reshape(len(ELEVS), N_AZ).copy()
    r[2, 1] = 2.0 / C.MISSION_RANGE                       # elevation 0, azimuth +22.5 deg (to the left)
    obs[M_RANGES] = r.flatten()
    a = potential_field(obs)
    assert a[1] < 0 and a[0] > 0


def test_bendy_ruler_climbs_over_a_low_wide_wall():
    obs = make_obs(blocked=[(90, {-15.0, 0.0, 15.0}, 6.0)])   # 35 deg up is clear, sideways is blocked to 90 deg
    a = BendyRuler()(obs)
    assert a[2] > 0 and a[0] > 0 and np.linalg.norm(a) == pytest.approx(CRUISE)


def test_bendy_ruler_goes_around_a_tall_narrow_obstacle():
    obs = make_obs(blocked=[(25, {-15.0, 0.0, 15.0, 35.0}, 6.0)])
    a = BendyRuler()(obs)
    assert abs(a[1]) > 0.3 and a[0] > 0


def test_bendy_ruler_slows_down_when_everything_is_blocked():
    obs = make_obs(blocked=[(180, {-35.0, -15.0, 0.0, 15.0, 35.0}, 4.0)])
    assert np.linalg.norm(BendyRuler()(obs)) < 0.35


def test_bendy_ruler_flies_straight_when_the_way_is_clear():
    assert BendyRuler()(make_obs()) == pytest.approx(straight_line(make_obs()))


def test_supervisor_hands_over_and_back_with_hysteresis():
    sup = Supervisor(lambda obs: np.array([0.0, 0.7, 0.0]), hold_s=0.1)
    sup.reset()
    assert not sup.guided and sup(make_obs()) == pytest.approx(straight_line(make_obs()))
    blocked = make_obs(blocked=[(30, {0.0}, 8.0)])
    assert sup(blocked) == pytest.approx([0.0, 0.7, 0.0]) and sup.guided
    half_clear = make_obs(blocked=[(30, {0.0}, 13.0)])                # over the trigger, under trigger + margin
    sup(half_clear)
    assert sup.guided
    for _ in range(3):                                                # hold_s = 0.1 s = 3 control steps
        sup(make_obs())
    assert not sup.guided
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_mission_policies.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'drone_follow.mission_policies'`

- [ ] **Step 3: Create `drone_follow/mission_policies.py`**

```python
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
    `lookahead` metres. Fly the open one closest to the waypoint direction, keeping the previous choice unless
    another is more than `hysteresis_deg` closer, so it does not flip between two gaps."""

    def __init__(self, lookahead: float = 10.0, hysteresis_deg: float = 10.0, min_elev: float = -15.0):
        self.lookahead, self.hysteresis_deg, self.min_elev = lookahead, hysteresis_deg, min_elev
        self.reset()

    def reset(self):
        self._last = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        r = ranges(obs).reshape(len(ELEVS), N_AZ)
        free = np.minimum(r, np.minimum(np.roll(r, 1, axis=1), np.roll(r, -1, axis=1))).flatten()
        allowed = RAY_ELEV >= self.min_elev
        open_ = allowed & (free >= self.lookahead)
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_mission_policies.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add drone_follow/mission_policies.py tests/test_mission_policies.py
git commit -m "Mission avoiders: straight line, potential field, BendyRuler-style, AUTO/GUIDED supervisor

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Seeded evaluation and results table

**Files:**
- Create: `drone_follow/mission_evaluate.py`
- Test: `tests/test_mission_evaluate.py`

**Interfaces:**
- Consumes: `make_mission_env` (Task 3); `straight_line`, `potential_field`, `BendyRuler`, `Supervisor`, `load_ppo` (Task 4).
- Produces:
  - `EVAL_SEED0 = 30_000`.
  - `run_legs(policy, level: int, legs: int, seed0: int = EVAL_SEED0, length: float | None = None) -> dict` with keys `level, legs, crash_rate, success_rate, failure_rate, extra_distance (mean over successes, fraction), time_ratio (mean over successes), guided_fraction, mean_action_change, p95_accel_ms2, crash_kinds (dict), failures (dict)`.
  - `make_policy(name: str, model: str | None = None)` for names `straight`, `potential`, `bendy`, `rl`, `rl_alone`.
  - `markdown_table(results: list[dict]) -> str` (each dict = a `run_legs` result plus `policy`).
  - CLI: `python -m drone_follow.mission_evaluate --policy bendy --level 2 --legs 500 --out results/mission/bendy_L2.json` and `python -m drone_follow.mission_evaluate --table results/mission/*.json`.

- [ ] **Step 1: Write the failing tests** (`tests/test_mission_evaluate.py`)

```python
from drone_follow.mission_evaluate import make_policy, markdown_table, run_legs


def test_straight_line_succeeds_on_empty_short_legs():
    r = run_legs(make_policy("straight"), level=0, legs=2, length=25.0)
    assert r["legs"] == 2 and r["success_rate"] == 1.0 and r["crash_rate"] == 0.0
    assert 0.0 <= r["extra_distance"] < 0.15 and r["guided_fraction"] == 0.0


def test_supervised_avoider_reports_guided_time_and_outcomes_add_up():
    r = run_legs(make_policy("bendy"), level=3, legs=2, length=60.0)
    assert r["crash_rate"] + r["success_rate"] + r["failure_rate"] == 1.0
    assert r["guided_fraction"] > 0.0


def test_markdown_table_lists_every_result():
    rows = [dict(policy="bendy", level=2, legs=500, crash_rate=0.04, success_rate=0.9, failure_rate=0.06,
                 extra_distance=0.12, time_ratio=1.3, guided_fraction=0.2, mean_action_change=0.05,
                 p95_accel_ms2=3.0, crash_kinds={"tree": 20}, failures={"timeout": 30})]
    table = markdown_table(rows)
    assert "| bendy | 2 | 4.0% | 90.0% |" in table and table.count("\n") == 2
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_mission_evaluate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'drone_follow.mission_evaluate'`

- [ ] **Step 3: Create `drone_follow/mission_evaluate.py`**

```python
"""Score avoiders on fixed, seeded mission legs (the same legs for every policy).

    python -m drone_follow.mission_evaluate --policy bendy --level 2 --legs 500 --out results/mission/bendy_L2.json
    python -m drone_follow.mission_evaluate --table results/mission/*.json

Heavy (500 legs take ~40 min per policy and level): run on gpu-sys, never the laptop.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

from drone_follow import config as C
from drone_follow.mission_env import make_mission_env
from drone_follow.mission_policies import BendyRuler, Supervisor, load_ppo, potential_field, straight_line

EVAL_SEED0 = 30_000                      # distinct from training seeds (seed * 1000 + i)
POLICIES = ("straight", "potential", "bendy", "rl", "rl_alone")


def make_policy(name: str, model: str | None = None):
    if name == "straight":
        return straight_line
    if name == "potential":
        return Supervisor(potential_field)
    if name == "bendy":
        return Supervisor(BendyRuler())
    if name in ("rl", "rl_alone"):
        if not model:
            raise ValueError(f"policy {name} needs --model")
        ppo = load_ppo(model)
        return Supervisor(ppo) if name == "rl" else ppo
    raise ValueError(f"unknown policy {name}")


def run_legs(policy, level: int, legs: int, seed0: int = EVAL_SEED0, length: float | None = None) -> dict:
    env = make_mission_env(level=level, seed=seed0, length=length)()
    sim = env.unwrapped
    crashes = successes = 0
    extra, time_ratio, guided, action_change, accel = [], [], [], [], []
    kinds, failures = Counter(), Counter()
    for ep in range(legs):
        obs, _ = env.reset(seed=seed0 + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        done, steps, guided_steps, prev_a, prev_v = False, 0, 0, None, None
        while not done:
            a = np.clip(np.asarray(policy(obs), float).reshape(3), -1.0, 1.0)
            guided_steps += int(getattr(policy, "guided", False))
            obs, _, term, trunc, info = env.step(a)
            v = sim.vel[0].copy()
            if prev_a is not None:
                action_change.append(float(np.linalg.norm(a - prev_a)))
                accel.append(float(np.linalg.norm(v - prev_v)) * C.CTRL_FREQ)
            prev_a, prev_v = a, v
            steps += 1
            done = term or trunc
        guided.append(guided_steps / steps)
        if info["collision"]:
            crashes += 1
            kinds[info["collision_kind"]] += 1
        elif info["success"]:
            successes += 1
            extra.append(info["path_m"] / info["leg_m"] - 1.0)
            time_ratio.append(info["time_s"] / (info["leg_m"] / C.MISSION_CRUISE))
        else:
            failures[info["failure"]] += 1
    env.close()                          # one world per call, closed here: nothing accumulates across calls
    return {"level": level, "legs": legs,
            "crash_rate": crashes / legs, "success_rate": successes / legs,
            "failure_rate": (legs - crashes - successes) / legs,
            "extra_distance": float(np.mean(extra)) if extra else None,
            "time_ratio": float(np.mean(time_ratio)) if time_ratio else None,
            "guided_fraction": float(np.mean(guided)),
            "mean_action_change": float(np.mean(action_change)) if action_change else 0.0,
            "p95_accel_ms2": float(np.percentile(accel, 95)) if accel else 0.0,
            "crash_kinds": dict(kinds), "failures": dict(failures)}


def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def markdown_table(results: list) -> str:
    rows = sorted(results, key=lambda r: (r["level"], r["crash_rate"], -r["success_rate"]))
    out = ["| policy | level | crashes | success | extra distance | time vs straight | guided | crashes by cause |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = "-" if r["time_ratio"] is None else f"{r['time_ratio']:.2f}x"
        out.append(f"| {r['policy']} | {r['level']} | {_pct(r['crash_rate'])} | {_pct(r['success_rate'])} | "
                   f"{_pct(r['extra_distance'])} | {t} | {_pct(r['guided_fraction'])} | {r['crash_kinds'] or '-'} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=POLICIES)
    ap.add_argument("--model", default=None, help="PPO .zip for rl / rl_alone")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--legs", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--out", default=None, help="write the result to this JSON file")
    ap.add_argument("--table", nargs="*", default=None, help="print a markdown table of these result files")
    args = ap.parse_args()
    if args.table is not None:
        print(markdown_table([json.load(open(p)) for p in args.table]))
        return
    r = run_legs(make_policy(args.policy, args.model), args.level, args.legs, args.seed0)
    r["policy"] = args.policy if not args.model else f"{args.policy}:{os.path.basename(os.path.dirname(args.model))}"
    print(json.dumps(r), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_mission_evaluate.py -q`
Expected: 3 passed (the level-3 test flies two 60 m legs, about 10-20 s).

- [ ] **Step 5: Commit**

```bash
git add drone_follow/mission_evaluate.py tests/test_mission_evaluate.py
git commit -m "Mission evaluation: seeded legs, crash/success/detour metrics, markdown table

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: PPO training for the RL avoider

**Files:**
- Create: `drone_follow/mission_train.py`
- Test: `tests/test_mission_train.py`

**Interfaces:**
- Consumes: `make_mission_env` (Task 3); `run_legs` (Task 5); `ProgressWriter`, `LatestSaver`, `TimeBudget` from `drone_follow/train_long.py`; `CollisionRate` from `drone_follow/train.py`.
- Produces: `MISSION_MIX = [0, 1, 1, 2, 2, 3, 3]`; `MissionEval(out, every, legs, levels=(1, 2, 3))` callback writing `best.zip`, `best.json`, `latest.zip`, `eval_log.jsonl` (score = mean crash rate + 0.5 x mean (1 - success rate), lower is better); CLI `python -m drone_follow.mission_train --out runs/mission_s1 --seed 1`.

- [ ] **Step 1: Write the failing test** (`tests/test_mission_train.py`)

```python
import json

from drone_follow import mission_train


class FakeModel:
    class logger:
        @staticmethod
        def record(*a, **k):
            pass

    def __init__(self):
        self.saved = []

    def predict(self, obs, deterministic=True):
        return obs[:3] * 0, None

    def save(self, path):
        self.saved.append(path)


def test_eval_keeps_the_best_score_and_logs_every_eval(tmp_path, monkeypatch):
    state = {}

    def fake_run_legs(policy, level, legs):
        crash, success = state["now"]
        return {"crash_rate": crash, "success_rate": success}

    monkeypatch.setattr(mission_train, "run_legs", fake_run_legs)
    cb = mission_train.MissionEval(str(tmp_path), every=1, legs=3, levels=(1, 2))
    cb.model = FakeModel()
    for outcome in [(0.5, 0.4), (0.2, 0.7), (0.3, 0.6)]:         # (crash, success) per eval, all levels
        state["now"] = outcome
        cb.run_eval()
    best = json.loads((tmp_path / "best.json").read_text())
    log = [json.loads(l) for l in (tmp_path / "eval_log.jsonl").read_text().splitlines()]
    assert best["L1_crash"] == 0.2 and len(log) == 3 and [e["best"] for e in log] == [True, True, False]
    assert sum(p.endswith("best.zip") for p in cb.model.saved) == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mission_train.py -q`
Expected: FAIL with `ImportError` / `ModuleNotFoundError: No module named 'drone_follow.mission_train'`

- [ ] **Step 3: Create `drone_follow/mission_train.py`**

```python
"""PPO on mission legs: the RL avoider. Meant for gpu-sys (never the laptop).

    python -m drone_follow.mission_train --out runs/mission_s1 --seed 1 --total-steps 5000000

The policy learns whole legs (it is not wrapped in the supervisor during training); mission_evaluate scores
it both alone (`rl_alone`) and under the supervisor (`rl`). Every --eval-every steps it scores itself on fixed
legs and keeps best.zip by crash rate, then success rate.
"""
import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from drone_follow.mission_env import make_mission_env
from drone_follow.mission_evaluate import run_legs
from drone_follow.train import CollisionRate
from drone_follow.train_long import LatestSaver, ProgressWriter, TimeBudget

MISSION_MIX = [0, 1, 1, 2, 2, 3, 3]          # levels cycled over the parallel envs
EVAL_LEVELS = (1, 2, 3)


class MissionEval(BaseCallback):
    """Scores the policy alone on fixed legs; keeps the safest checkpoint (score: lower is better)."""

    def __init__(self, out: str, every: int, legs: int, levels=EVAL_LEVELS):
        super().__init__()
        self.out, self.every, self.legs, self.levels = out, every, legs, levels
        self.last, self.best = 0, np.inf
        try:                                    # resuming: keep the best score from before
            with open(os.path.join(out, "best.json")) as f:
                self.best = float(json.load(f)["score"])
        except (OSError, ValueError, KeyError):
            pass

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self.run_eval()
        return True

    def run_eval(self):
        policy = lambda obs: self.model.predict(obs, deterministic=True)[0]
        metrics = {}
        for lvl in self.levels:
            r = run_legs(policy, lvl, self.legs)          # builds and closes its own env: no leak across evals
            metrics[f"L{lvl}_crash"] = r["crash_rate"]
            metrics[f"L{lvl}_success"] = r["success_rate"]
        score = float(np.mean([metrics[f"L{l}_crash"] for l in self.levels])
                      + 0.5 * np.mean([1.0 - metrics[f"L{l}_success"] for l in self.levels]))
        for k, v in metrics.items():
            self.logger.record(f"eval/{k}", v)
        self.model.save(os.path.join(self.out, "latest.zip"))
        improved = bool(score < self.best)
        if improved:
            self.best = score
            self.model.save(os.path.join(self.out, "best.zip"))
            with open(os.path.join(self.out, "best.json"), "w") as f:
                json.dump({"steps": self.num_timesteps, "score": score, **metrics}, f, indent=2)
        with open(os.path.join(self.out, "eval_log.jsonl"), "a") as f:
            f.write(json.dumps({"steps": self.num_timesteps, "score": score, "best": improved, **metrics}) + "\n")
        print(f"[{self.num_timesteps}] eval {metrics} score={score:.3f}{' *best*' if improved else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--envs", type=int, default=7)
    ap.add_argument("--net", type=int, default=256)
    ap.add_argument("--total-steps", type=int, default=5_000_000)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--eval-every", type=int, default=1_000_000)
    ap.add_argument("--eval-legs", type=int, default=20)
    ap.add_argument("--lr-start", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=2e-5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)

    def lr_schedule(progress_remaining: float) -> float:
        return args.lr_end + (args.lr_start - args.lr_end) * progress_remaining

    venv = VecMonitor(SubprocVecEnv([make_mission_env(level=MISSION_MIX[i % len(MISSION_MIX)],
                                                      seed=args.seed * 1000 + i) for i in range(args.envs)]))
    n_steps = 2048
    batch = min(2048, n_steps * args.envs // 4)
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu", custom_objects={"learning_rate": lr_schedule})
    else:
        model = PPO(
            "MlpPolicy", venv, n_steps=n_steps, batch_size=batch, n_epochs=10,
            learning_rate=lr_schedule, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, max_grad_norm=0.5, seed=args.seed, device="cpu",
            policy_kwargs=dict(net_arch=dict(pi=[args.net, args.net], vf=[args.net, args.net]), log_std_init=-1.0),
            tensorboard_log=os.path.join(args.out, "tb"), verbose=1)
    print(f"config: {vars(args)}  rollout={n_steps * args.envs} batch={batch}", flush=True)

    evaluator = MissionEval(args.out, args.eval_every, args.eval_legs)
    callbacks = [CollisionRate(), evaluator, LatestSaver(args.out),
                 ProgressWriter(args.out, args.total_steps, args.max_hours), TimeBudget(args.max_hours)]
    model.learn(max(args.total_steps - model.num_timesteps, 1), reset_num_timesteps=False,
                callback=callbacks, tb_log_name="mission")
    model.save(os.path.join(args.out, "final"))
    evaluator.run_eval()
    venv.close()
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test and the whole suite** (no training runs here: the test fakes `run_legs` and the model)

Run: `.venv/bin/python -m pytest tests/test_mission_train.py -q && .venv/bin/python -m pytest tests -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add drone_follow/mission_train.py tests/test_mission_train.py
git commit -m "Mission PPO trainer with seeded eval callback (best by crash, then success)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Runs on gpu-sys and the results table

All commands run from the Mac repo root. `S` is the session scratchpad directory. **Ask the user before Step 1 (copying code) and before Steps 3 and 5 (launching jobs).**

**Files:**
- Modify: `README.md` (new section "Waypoint missions")
- Results: `results/mission/*.json` (brought back from gpu-sys)

- [ ] **Step 1: Copy the committed code to gpu-sys** (tracked files of `drone_follow/` and `tests/` only; `runs/` is untouched)

```bash
git archive --format=tar.gz HEAD drone_follow tests | base64 > $S/code.b64
cat $S/code.b64 | ssh -o ConnectTimeout=20 gpu-sys 'wsl -e bash -c "cat > ~/code.b64"'
cat > $S/unpack.sh <<'EOF'
cd ~/DetectionAndAvoidance && base64 -d ~/code.b64 > ~/code.tgz && tar -xzf ~/code.tgz && rm ~/code.b64 ~/code.tgz
export PYTHONPATH=$HOME/DetectionAndAvoidance:$HOME/DetectionAndAvoidance/vendor/gym-pybullet-drones
~/bin/micromamba run -n df python -m pytest tests/test_mission_world.py tests/test_mission_policies.py tests/test_mission_train.py -q 2>&1 | tail -3
EOF
ssh -o ConnectTimeout=20 gpu-sys 'wsl -e bash -s' < $S/unpack.sh
```
Expected: the last line reads `N passed`.

- [ ] **Step 2: Write the two job scripts and copy them over**

```bash
cat > $S/mission_eval_job.sh <<'EOF'
#!/bin/bash
# Usage: mission_eval_job.sh "<policy[:model]> ..." -- scores each policy on levels 0-3, 500 legs, 12 at a time
cd ~/DetectionAndAvoidance
export PYTHONPATH=$HOME/DetectionAndAvoidance:$HOME/DetectionAndAvoidance/vendor/gym-pybullet-drones
mkdir -p results/mission logs
for spec in $1; do
  pol=${spec%%:*}; model=${spec#*:}; [ "$model" = "$spec" ] && model=""
  tag=$pol; [ -n "$model" ] && tag=${pol}_$(basename $(dirname $model))
  for lvl in 0 1 2 3; do
    echo "$pol $lvl ${model:-none} results/mission/${tag}_L${lvl}.json"
  done
done | xargs -P 12 -L 1 bash -c '~/bin/micromamba run -n df python -m drone_follow.mission_evaluate --policy $0 --level $1 --legs 500 $( [ "$2" != none ] && echo --model $2 ) --out $3 > logs/$(basename $3 .json).log 2>&1'
echo done > results/mission/DONE_$(date +%s)
EOF
cat > $S/mission_train_job.sh <<'EOF'
#!/bin/bash
cd ~/DetectionAndAvoidance
export PYTHONPATH=$HOME/DetectionAndAvoidance:$HOME/DetectionAndAvoidance/vendor/gym-pybullet-drones
for seed in 1 2; do
  mkdir -p runs/mission_s$seed
  ~/bin/micromamba run -n df python -u -m drone_follow.mission_train --out runs/mission_s$seed --seed $seed \
      --total-steps 5000000 --envs 7 --threads 2 > runs/mission_s$seed/train.log 2>&1 &
done
wait
EOF
for f in mission_eval_job.sh mission_train_job.sh; do
  cat $S/$f | ssh -o ConnectTimeout=20 gpu-sys "wsl -e bash -c \"cat > ~/$f\""
done
```

- [ ] **Step 3: Launch the baseline evaluation and the RL training together** (12 eval processes + 2 learners with 7 workers each slightly oversubscribe 16 cores; the evals finish in about an hour)

The eval policies are passed through a wrapper script because the WMI command line cannot carry quotes reliably:

```bash
echo 'bash ~/mission_eval_job.sh "straight potential bendy"' > $S/run_baselines.sh
cat $S/run_baselines.sh | ssh -o ConnectTimeout=20 gpu-sys 'wsl -e bash -c "cat > ~/run_baselines.sh"'
for job in run_baselines.sh mission_train_job.sh; do
  ssh -o ConnectTimeout=20 gpu-sys "powershell -NoProfile -Command \"(Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='wsl.exe -e bash /home/aibel_win/$job'})\""
done
```
Expected: two `ReturnValue 0` lines.

- [ ] **Step 4: Check progress with short ssh calls** (every 20-30 min; never a long foreground ssh)

```bash
cat > $S/mission_status.sh <<'EOF'
cd ~/DetectionAndAvoidance
ls results/mission/ 2>/dev/null
for s in 1 2; do cut -c1-200 runs/mission_s$s/progress.json 2>/dev/null; echo; tail -n 2 runs/mission_s$s/eval_log.jsonl 2>/dev/null; done
free -g | head -2
EOF
ssh -o ConnectTimeout=20 gpu-sys 'wsl -e bash -s' < $S/mission_status.sh
```
Watch `rss_mb` in each `progress.json`: it must stay roughly flat (about 400-600 MB). If it climbs by hundreds of MB per eval, stop the runs (`pkill -f mission_train`) and investigate before continuing. Training ends when both `train.log` files print `training complete` (roughly 3-4 h).

- [ ] **Step 5: Score both RL seeds** (after training completes; ask the user first)

```bash
echo 'bash ~/mission_eval_job.sh "rl:runs/mission_s1/best.zip rl:runs/mission_s2/best.zip rl_alone:runs/mission_s1/best.zip rl_alone:runs/mission_s2/best.zip"' > $S/run_rl_eval.sh
cat $S/run_rl_eval.sh | ssh -o ConnectTimeout=20 gpu-sys 'wsl -e bash -c "cat > ~/run_rl_eval.sh"'
ssh -o ConnectTimeout=20 gpu-sys "powershell -NoProfile -Command \"(Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='wsl.exe -e bash /home/aibel_win/run_rl_eval.sh'})\""
```
Poll with Step 4's script until a second `DONE_*` file appears in `results/mission/`.

- [ ] **Step 6: Bring the results and run logs back to the Mac**

```bash
echo 'cd ~/DetectionAndAvoidance && tar -cz results/mission runs/mission_s1/eval_log.jsonl runs/mission_s1/best.json runs/mission_s2/eval_log.jsonl runs/mission_s2/best.json | base64 -w0' > $S/fetch.sh
ssh -o ConnectTimeout=20 gpu-sys 'wsl -e bash -s' < $S/fetch.sh | base64 -d | tar -xz
ls results/mission
```
Expected: 28 JSON files (3 baselines x 4 levels + 2 RL modes x 2 seeds x 4 levels) and two `DONE_*` markers.

- [ ] **Step 7: Add the table to the README and commit**

```bash
.venv/bin/python -m drone_follow.mission_evaluate --table results/mission/*.json
```
Add a `## Waypoint missions` section to `README.md` with: one sentence on the task (legs of 80-150 m at 5 m/s, levels 0-3), the printed table, one sentence per finding (which avoider has the fewest crashes per level, whether RL beats BendyRuler, and the noise level: at 500 legs a rate near 10% carries about ±2.7 points at 95%). Then:

`runs/` is git-ignored, so keep the training eval logs next to the results:

```bash
rm -f results/mission/DONE_*
for s in 1 2; do cp runs/mission_s$s/eval_log.jsonl results/mission/train_s${s}_eval_log.jsonl; cp runs/mission_s$s/best.json results/mission/train_s${s}_best.json; done
git add README.md results/mission
git commit -m "Waypoint missions: baseline and RL avoider results on 500 seeded legs per level

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
