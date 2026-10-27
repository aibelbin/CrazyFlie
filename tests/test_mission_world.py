import numpy as np
import pybullet as p
import pytest

from drone_follow import config as C
from drone_follow.mission_world import Leg, MissionTerrain, attach_leg, make_leg


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


def test_closest_points_are_as_far_as_the_gaps():
    leg = make_leg(np.random.default_rng(2), 3, length=100.0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        pos = rng.uniform([0, -10, 0], [100, 10, 15])
        g, _ = leg.gaps(pos)
        pts = leg.closest_points(pos)
        assert pts.shape == (len(g), 3)
        assert np.linalg.norm(pts - pos, axis=1) == pytest.approx(np.maximum(g, 0.0), abs=1e-9)
    pole = Leg(10.0, np.zeros(3), np.zeros(3), leg.terrain, cylinders=np.array([[0.0, 0.0, 0.0, 8.0, 0.15]]))
    assert pole.closest_points(np.array([3.0, 4.0, 2.0]))[0] == pytest.approx([0.09, 0.12, 2.0])
    assert pole.closest_points(np.array([0.05, 0.0, 10.0]))[0] == pytest.approx([0.05, 0.0, 8.0])   # over the top


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
