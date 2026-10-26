"""The simulated world around the drone: a walking person, debris, and a rival drone.

State lives in NumPy so a scenario can be sampled before PyBullet exists; `attach`
creates matching PyBullet bodies and `_sync` mirrors positions into them each step.
Debris and the rival drone are spheres (so ray casts and the analytic collision
check agree). The person is a visual-only capsule, so rays and physics ignore it.
"""
from dataclasses import dataclass

import numpy as np
import pybullet as p

from drone_follow import config as C
from drone_follow.walker import Walker, build_scenery
from drone_follow.world import Terrain, forest

DEBRIS_RGBA = (0.55, 0.5, 0.45, 1.0)
RIVAL_RGBA = (0.9, 0.1, 0.1, 1.0)
PEDESTRIAN_RGBA = (0.95, 0.55, 0.1, 1.0)
PEDESTRIAN_SPHERE_Z = (0.3, 0.9, 1.5)     # a walking person = a column of three spheres
PEDESTRIAN_RADIUS = 0.3
PERSON_RGBA = (0.1, 0.8, 0.3, 1.0)
SHIRTS = [(0.9, 0.5, 0.1, 1.0), (0.7, 0.2, 0.6, 1.0), (0.9, 0.85, 0.2, 1.0), (0.2, 0.6, 0.8, 1.0)]
ARENA = 6.0


@dataclass
class Obstacle:
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    kind: str                      # "debris" | "rival" | "pedestrian"
    active: bool = True
    body: int = -1
    group: int = -1                # pedestrians: the three spheres of one person share a group
    z_off: float = 0.0             # pedestrians: sphere height above the ground
    fall: bool = False             # falling debris: accelerates downward until it lands


class Scenario:
    DEFER_SYNC = False        # live demo: the caller interpolates visuals between control steps

    def __init__(self, rng: np.random.Generator, difficulty: int, pedestrians: bool = True,
                 humanoid: bool = False):
        """difficulty 0: clear sky, 1: static debris, 2: drifting debris + a rival drone,
        3: stress test (denser debris, two faster rival drones).
        Levels >= 2 (when `pedestrians` is on) also add people walking across the arena."""
        self.rng = rng
        self.difficulty = difficulty
        # levels 4-5 add hills, a forest and (on the F450) wind: 4 = level-2 content, 5 = level-3 content + falling debris
        self.terrain = Terrain(rng) if difficulty >= 4 else None
        eff = {4: 2, 5: 3}.get(difficulty, difficulty)
        self.half = 16.0 if self.terrain else 5.0                 # half-width of the area obstacles are placed in
        self.person_range = C.WORLD_PERSON if self.terrain else C.ARENA_PERSON
        self.bound = C.WORLD_BOUND if self.terrain else C.ARENA_BOUND
        self.z_cap = C.WORLD_Z_MAX if self.terrain else C.Z_MAX
        self._eff = eff
        self.humanoid = humanoid            # draw walking figures + scenery (visual only)
        self._walkers: dict = {}
        self._next_group = 0
        self.t = 0.0
        self.client = None
        self.person_body = -1
        self.obstacles: list[Obstacle] = []

        self.person = rng.uniform(-self.person_range, self.person_range, 2)
        self.person_vel = np.zeros(2)
        self._speed = rng.uniform(0.3, 0.7)
        self._waypoint = self._new_waypoint()
        self.external_person = False        # True when a webcam drives the person
        self._prev_person = self.person.copy()
        self._prev_pos: dict = {}

        ang = rng.uniform(0, 2 * np.pi)
        start_xy = self.person + 2.5 * np.array([np.cos(ang), np.sin(ang)])
        self.drone_start = np.array([start_xy[0], start_xy[1], C.FOLLOW_ALT + self.ground(start_xy)])

        self.static_pos, self.static_r = np.zeros((0, 3)), np.zeros(0)
        self.static_kind, self.static_group = np.array([], dtype=object), np.zeros(0, dtype=int)
        if self.terrain:
            keep = [(self.person, 2.5), (start_xy, 2.5)]
            self.static_pos, self.static_r, self.static_kind, self.static_group = forest(rng, self.terrain, keep)
        if difficulty >= 1:
            self._place_debris()
        if pedestrians and difficulty >= 2:
            for _ in range(3 if eff >= 3 else rng.integers(1, 3)):
                self.add_pedestrian()
        n_rivals = 0 if difficulty < 2 else (2 if eff >= 3 else int(rng.random() < 0.8))
        self._rival_times = sorted(rng.uniform(1.0, 7.0) for _ in range(n_rivals))
        self._fall_times = sorted(rng.uniform(2.0, 16.0) for _ in range(int(rng.integers(2, 5)))) if difficulty >= 5 else []

    # ------------------------------------------------------------------ queries
    def ground(self, xy) -> float:
        """Terrain height under (x, y); 0 on the flat levels."""
        return 0.0 if self.terrain is None else float(self.terrain.height(xy[0], xy[1]))

    @property
    def target_point(self) -> np.ndarray:
        return np.array([self.person[0], self.person[1], C.FOLLOW_ALT + self.ground(self.person)])

    def live_obstacles(self):
        return [o for o in self.obstacles if o.active]

    def clearance(self, pos: np.ndarray) -> float:
        """Smallest surface-to-drone-body gap over all active obstacles (inf if none)."""
        best = np.inf
        for o in self.live_obstacles():
            best = min(best, np.linalg.norm(pos - o.pos) - o.radius - C.DRONE_RADIUS)
        if len(self.static_r):
            best = min(best, float((np.linalg.norm(self.static_pos - pos, axis=1) - self.static_r).min() - C.DRONE_RADIUS))
        return float(best)

    def static_hazard(self, pos: np.ndarray, vel: np.ndarray):
        """(urgency, min gap) over the static trees and rocks, vectorised (they do not move)."""
        if not len(self.static_r):
            return 0.0, np.inf
        rel = self.static_pos - pos
        dist = np.linalg.norm(rel, axis=1) + 1e-9
        gap = dist - self.static_r - C.DRONE_RADIUS
        closing = (rel / dist[:, None]) @ vel                       # > 0 when flying toward it
        ok = closing > 0.05
        urgency = 0.0
        if ok.any():
            ttc = np.maximum(gap[ok], 0.0) / closing[ok]
            urgency = float((np.clip((C.TTC_START - ttc) / C.TTC_START, 0.0, 1.0) ** 2).max())
        return urgency, float(gap.min())

    def nearest_static_kind(self, pos: np.ndarray):
        if not len(self.static_r):
            return None, np.inf
        gaps = np.linalg.norm(self.static_pos - pos, axis=1) - self.static_r
        i = int(np.argmin(gaps))
        return str(self.static_kind[i]), float(gaps[i])

    # ----------------------------------------------------------------- spawning
    def _new_waypoint(self) -> np.ndarray:
        return self.rng.uniform(-self.person_range, self.person_range, 2)

    def _place_debris(self):
        rng = self.rng
        start, goal = self.drone_start, self.target_point
        axis = goal - start
        side = np.array([-axis[1], axis[0], 0.0])
        side /= np.linalg.norm(side) + 1e-9
        for _ in range(rng.integers(1, 3)):                 # blocking the straight path
            frac = rng.uniform(0.3, 0.7)
            pos = start + frac * axis + side * rng.uniform(-0.25, 0.25)
            pos[2] += rng.uniform(-0.2, 0.2)
            self.obstacles.append(
                Obstacle(pos, np.zeros(3), rng.uniform(0.3, 0.5), "debris")
            )
        n_scatter = rng.integers(3, 6) if self._eff >= 3 else rng.integers(1, 4)
        for _ in range(n_scatter):                          # scattered around
            for _try in range(20):
                xy = rng.uniform(-self.half, self.half, 2)
                pos = np.array([xy[0], xy[1], rng.uniform(0.5, 2.2) + self.ground(xy)])
                r = rng.uniform(0.25, 0.5)
                if (np.linalg.norm(pos - start) > 1.2 + r
                        and np.linalg.norm(pos[:2] - self.person) > 1.5 + r):
                    break
            vel = np.zeros(3)
            if self._eff >= 2 and rng.random() < 0.6:
                a = rng.uniform(0, 2 * np.pi)
                vel = rng.uniform(0.2, 0.6) * np.array([np.cos(a), np.sin(a), rng.uniform(-0.1, 0.1)])
            self.obstacles.append(Obstacle(pos, vel, r, "debris"))

    def add_pedestrian(self, xy=None, vel_xy=None):
        """A person walking in a straight line (bouncing off the arena edge)."""
        rng = self.rng
        if xy is None:
            for _try in range(20):
                xy = rng.uniform(-self.half, self.half, 2)
                if np.linalg.norm(xy - self.drone_start[:2]) > 1.8:
                    break
        xy = np.asarray(xy, float)
        if vel_xy is None:
            a = rng.uniform(0, 2 * np.pi)
            vel_xy = rng.uniform(0.4, 1.0) * np.array([np.cos(a), np.sin(a)])
        added = []
        group, self._next_group = self._next_group, self._next_group + 1
        for z in PEDESTRIAN_SPHERE_Z:
            ob = Obstacle(np.array([xy[0], xy[1], z + self.ground(xy)]), np.array([vel_xy[0], vel_xy[1], 0.0]),
                          PEDESTRIAN_RADIUS, "pedestrian", group=group, z_off=z)
            self.obstacles.append(ob)
            if self.client is not None:
                self._make_body(ob)
            added.append(ob)
        return added

    def spawn_rival(self, drone_pos: np.ndarray, speed: float | None = None):
        """A rival drone appears ~6 m out and flies a straight line at the drone."""
        rng = self.rng
        az, el = rng.uniform(0, 2 * np.pi), rng.uniform(-0.25, 0.25)
        direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        pos = drone_pos + 6.0 * direction
        g = self.ground(pos[:2])
        pos[2] = np.clip(pos[2], g + 0.5, g + 2.8)
        aim = drone_pos + rng.uniform(-0.4, 0.4, 3)
        v = (aim - pos)
        lo, hi = (1.4, 2.4) if self._eff >= 3 else (0.8, 1.3)
        v = v / np.linalg.norm(v) * (speed if speed else rng.uniform(lo, hi))
        ob = Obstacle(pos, v, 0.2, "rival")
        self.obstacles.append(ob)
        if self.client is not None:
            self._make_body(ob)
        return ob

    def spawn_debris_ahead(self, drone_pos: np.ndarray):
        """Debris dropped on the line between the drone and the person (for demos)."""
        axis = self.target_point - drone_pos
        pos = drone_pos + 0.5 * axis
        ob = Obstacle(pos, np.zeros(3), self.rng.uniform(0.3, 0.45), "debris")
        self.obstacles.append(ob)
        if self.client is not None:
            self._make_body(ob)
        return ob

    def spawn_falling(self, drone_pos: np.ndarray, drone_vel: np.ndarray | None = None):
        """A branch or rock that drops out of the canopy ahead of the drone (sudden hazard)."""
        rng = self.rng
        d = np.zeros(2) if drone_vel is None else np.asarray(drone_vel, float)[:2]
        n = float(np.linalg.norm(d))
        d = d / n if n > 0.2 else (self.person - drone_pos[:2]) / (np.linalg.norm(self.person - drone_pos[:2]) + 1e-9)
        xy = drone_pos[:2] + 3.5 * d + rng.uniform(-0.4, 0.4, 2)
        ob = Obstacle(np.array([xy[0], xy[1], self.ground(xy) + 3.6]), np.zeros(3), rng.uniform(0.25, 0.4),
                      "debris", fall=True)
        self.obstacles.append(ob)
        if self.client is not None:
            self._make_body(ob)
        return ob

    # --------------------------------------------------------------- simulation
    def set_person(self, xy: np.ndarray, dt: float):
        """Drive the person externally (webcam). Velocity is finite-differenced."""
        xy = np.asarray(xy, float)
        self.person_vel = 0.7 * self.person_vel + 0.3 * (xy - self.person) / dt
        self.person = xy
        self.external_person = True

    def advance(self, dt: float, drone_pos: np.ndarray, drone_vel: np.ndarray | None = None):
        self.t += dt
        self._prev_person = self.person.copy()
        self._prev_pos = {id(o): o.pos.copy() for o in self.obstacles}
        if not self.external_person:
            to_wp = self._waypoint - self.person
            dist = np.linalg.norm(to_wp)
            if dist < 0.3:
                self._waypoint = self._new_waypoint()
                self._speed = self.rng.uniform(0.3, 0.7)
            else:
                self.person_vel = to_wp / dist * self._speed
                self.person = self.person + self.person_vel * dt
        while self._rival_times and self.t >= self._rival_times[0]:
            self._rival_times.pop(0)
            self.spawn_rival(drone_pos)
        while self._fall_times and self.t >= self._fall_times[0]:
            self._fall_times.pop(0)
            self.spawn_falling(drone_pos, drone_vel)
        edge = self.half + 1.0                                  # = ARENA on the flat levels
        for o in self.live_obstacles():
            if o.fall:
                o.vel[2] -= 9.8 * dt
            o.pos = o.pos + o.vel * dt
            g = self.ground(o.pos[:2])
            if o.fall and o.pos[2] - g < o.radius:               # landed: stays where it fell
                o.pos[2], o.vel, o.fall = g + o.radius, np.zeros(3), False
                continue
            if o.kind == "rival":
                if np.abs(o.pos[:2]).max() > self.bound + 1.0 or o.pos[2] - g < 0.1:
                    o.active = False
                continue
            for ax in (0, 1):
                if abs(o.pos[ax]) > edge:
                    o.vel[ax] *= -1
            if o.kind == "pedestrian":
                o.pos[2] = g + o.z_off
            elif o.kind == "debris" and not o.fall and np.any(o.vel) and not 0.4 < o.pos[2] - g < 2.6:
                o.vel[2] *= -1
        if not Scenario.DEFER_SYNC:
            self._sync()

    # ----------------------------------------------------------------- pybullet
    def attach(self, client: int):
        self.client = client
        self._walkers = {}
        if self.humanoid:
            build_scenery(client, self.rng)
            self._walkers["person"] = Walker(client, PERSON_RGBA)
        else:
            vis = p.createVisualShape(p.GEOM_CAPSULE, radius=0.22, length=1.2,
                                      rgbaColor=PERSON_RGBA, physicsClientId=client)
            self.person_body = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                basePosition=[*self.person, 0.82], physicsClientId=client)
        for o in self.obstacles:
            self._make_body(o)
        if self.terrain:
            self.terrain.attach(client)
            for pos, r in zip(self.static_pos, self.static_r):
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=float(r), physicsClientId=client)
                p.createMultiBody(0, col, -1, basePosition=pos.tolist(), physicsClientId=client)

    def _make_body(self, o: Obstacle):
        c = self.client
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=o.radius, physicsClientId=c)
        rgba = {"rival": RIVAL_RGBA, "pedestrian": PEDESTRIAN_RGBA}.get(o.kind, DEBRIS_RGBA)
        if o.kind == "pedestrian" and self.humanoid:
            rgba = (0, 0, 0, 0)               # the walking figure is drawn instead
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=o.radius, rgbaColor=rgba, physicsClientId=c)
        o.body = p.createMultiBody(0, col, vis, basePosition=o.pos.tolist(), physicsClientId=c)

    def sync_interp(self, alpha: float, dt: float):
        """Place visuals `alpha` of the way from the previous to the current control step."""
        self._sync(alpha, dt)

    def _at(self, o: Obstacle, alpha: float) -> np.ndarray:
        prev = self._prev_pos.get(id(o))
        return o.pos if prev is None or alpha >= 1.0 else prev + alpha * (o.pos - prev)

    def _sync(self, alpha: float = 1.0, dt: float = 1.0 / C.CTRL_FREQ):
        if self.client is None:
            return
        c = self.client
        person = self.person if alpha >= 1.0 else \
            self._prev_person + alpha * (self.person - self._prev_person)
        if self.humanoid:
            self._sync_walkers(person, alpha, dt)
        else:
            p.resetBasePositionAndOrientation(
                self.person_body, [*person, 0.82], [0, 0, 0, 1], physicsClientId=c)
        for o in self.obstacles:
            if o.body < 0:
                continue
            pos = self._at(o, alpha) if o.active else np.array([0.0, 0.0, -50.0])
            p.resetBasePositionAndOrientation(o.body, pos.tolist(), [0, 0, 0, 1], physicsClientId=c)

    def _sync_walkers(self, person: np.ndarray, alpha: float, dt: float):
        self._walkers["person"].update(person, self.person_vel, dt)
        for o in self.obstacles:
            if o.kind != "pedestrian" or o.body < 0 or abs(o.pos[2] - PEDESTRIAN_SPHERE_Z[0]) > 1e-6:
                continue                       # one walker per group, driven by its lowest sphere
            key = ("ped", o.group)
            if key not in self._walkers:
                self._walkers[key] = Walker(self.client, SHIRTS[o.group % len(SHIRTS)])
            self._walkers[key].update(self._at(o, alpha)[:2], o.vel[:2], dt)
