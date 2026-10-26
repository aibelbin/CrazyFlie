import numpy as np
import pybullet as p

from drone_follow import config as C
from drone_follow.env import make_env
from drone_follow.scenario import Scenario
from drone_follow.world import Terrain


def test_heightfield_matches_the_height_function():
    client = p.connect(p.DIRECT)
    try:
        t = Terrain(np.random.default_rng(5))
        t.attach(client)
        rng = np.random.default_rng(0)
        for _ in range(25):
            x, y = rng.uniform(-18, 18, 2)
            hit = p.rayTest([x, y, 60], [x, y, -5], physicsClientId=client)[0]
            assert abs((60 - 65 * hit[2]) - float(t.height(x, y))) < 0.03
    finally:
        p.disconnect(client)


def test_terrain_levels_place_the_drone_and_person_on_the_ground():
    for seed in range(15):
        s = Scenario(np.random.default_rng(seed), 4)
        assert s.terrain is not None and len(s.static_r) > 40
        assert abs(s.drone_start[2] - s.ground(s.drone_start[:2]) - C.FOLLOW_ALT) < 1e-9
        assert abs(s.target_point[2] - s.ground(s.person) - C.FOLLOW_ALT) < 1e-9
        for xy in (s.person, s.drone_start[:2]):                       # nothing grows on the start points
            assert np.linalg.norm(s.static_pos[:, :2] - xy, axis=1).min() > 2.5
        assert s.clearance(s.drone_start) > 0


def test_flat_levels_have_no_terrain():
    for lvl in range(4):
        s = Scenario(np.random.default_rng(1), lvl)
        assert s.terrain is None and len(s.static_r) == 0 and s.ground(np.zeros(2)) == 0.0


def test_level5_drops_debris_that_lands_on_the_ground():
    s = Scenario(np.random.default_rng(3), 5)
    assert len(s._fall_times) >= 2
    s.spawn_falling(s.drone_start, np.array([1.0, 0.0, 0.0]))
    ob = s.obstacles[-1]
    z0 = ob.pos[2] - s.ground(ob.pos[:2])
    assert ob.fall and z0 > 3.0
    for _ in range(90):
        s.advance(1 / 30, s.drone_start)
    assert not ob.fall and abs(ob.pos[2] - s.ground(ob.pos[:2]) - ob.radius) < 0.05
    assert not np.any(ob.vel)


def test_env_runs_on_terrain_levels_with_finite_observations():
    for lvl in (4, 5):
        env = make_env(difficulty=lvl, seed=2)()
        obs, _ = env.reset(seed=2)
        for _ in range(45):
            obs, r, term, trunc, info = env.step(np.zeros(3))
            assert np.isfinite(obs).all() and np.isfinite(r)
            if term or trunc:
                break
        env.close()
