import numpy as np
import pybullet as p
import pytest

from drone_follow import config as C
from drone_follow.env import OBS_DIM, FollowAvoidAviary, make_env
from drone_follow.policies import naive_follow, reactive_follow
from drone_follow.scenario import Obstacle, Scenario
from drone_follow.sensing import N_RAYS, range_scan


@pytest.fixture
def client():
    cid = p.connect(p.DIRECT)
    yield cid
    p.disconnect(cid)


def test_range_scan_sees_a_sphere(client):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5, physicsClientId=client)
    p.createMultiBody(0, col, basePosition=[3, 0, 1], physicsClientId=client)
    ranges = range_scan(client, np.array([0.0, 0.0, 1.0]))
    assert len(ranges) == N_RAYS
    assert ranges.min() == pytest.approx(2.5, abs=0.05)     # sphere surface at x = 2.5
    assert ranges.max() == pytest.approx(C.MAX_RANGE)


def test_scenario_is_reproducible_and_levels_differ():
    a = Scenario(np.random.default_rng(3), 2)
    b = Scenario(np.random.default_rng(3), 2)
    assert np.allclose(a.drone_start, b.drone_start)
    assert len(a.obstacles) == len(b.obstacles)
    assert len(Scenario(np.random.default_rng(3), 0).obstacles) == 0
    assert len(Scenario(np.random.default_rng(3), 1).obstacles) >= 2


def test_clearance_accounts_for_radii():
    s = Scenario(np.random.default_rng(0), 0)
    s.obstacles.append(Obstacle(np.array([2.0, 0, 1]), np.zeros(3), 0.5, "debris"))
    assert s.clearance(np.zeros(3) + [0, 0, 1]) == pytest.approx(2.0 - 0.5 - C.DRONE_RADIUS)


def test_env_shapes_and_finite_observations():
    env = make_env(difficulty=2, seed=0)()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    for _ in range(30):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
        if term or trunc:
            break
    env.close()


def test_same_seed_gives_same_episode():
    def rollout():
        env = make_env(difficulty=2, seed=5)()
        obs, _ = env.reset(seed=5)
        out = []
        for _ in range(20):
            obs, r, *_ = env.step(naive_follow(obs))
            out.append(r)
        env.close()
        return out
    assert rollout() == pytest.approx(rollout())


def test_velocity_command_moves_the_drone():
    env = make_env(difficulty=0, seed=1)()
    env.reset(seed=1)
    for _ in range(45):
        obs, *_ = env.step(np.array([1.0, 0.0, 0.0]))
    assert obs[7] * C.SPEED_LIMIT > 0.8       # forward speed reached after 1.5 s
    env.close()


def test_flying_into_debris_is_a_terminal_collision():
    env = make_env(difficulty=0, seed=2)()
    env.reset(seed=2)
    sim = env.unwrapped
    sim.scenario.obstacles.append(Obstacle(sim.pos[0] + [1.0, 0, 0], np.zeros(3), 0.4, "debris"))
    sim.scenario._make_body(sim.scenario.obstacles[-1])
    term = False
    for _ in range(60):
        _, r, term, _, info = env.step(np.array([1.0, 0.0, 0.0]))
        if term:
            break
    assert term and info["collision"] and r == C.R_COLLISION * C.REWARD_SCALE
    env.close()


def test_reactive_baseline_avoids_debris_that_naive_hits():
    def run(policy, seed):
        env = make_env(difficulty=1, seed=seed)()
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        env.close()
        return info["collision"]
    seeds = range(20, 30)
    naive = sum(run(naive_follow, s) for s in seeds)
    reactive = sum(run(reactive_follow, s) for s in seeds)
    assert reactive < naive


def test_pedestrians_appear_from_level_2_and_walk():
    assert not any(o.kind == "pedestrian" for o in Scenario(np.random.default_rng(1), 1).obstacles)
    s = Scenario(np.random.default_rng(1), 2)
    people = [o for o in s.obstacles if o.kind == "pedestrian"]
    assert len(people) % 3 == 0 and len(people) >= 3          # three spheres per person
    before = [o.pos.copy() for o in people]
    s.advance(1.0, np.array([0.0, 0.0, 1.2]))
    assert any(np.linalg.norm(o.pos - b) > 0.1 for o, b in zip(people, before))
    assert not any(o.kind == "pedestrian" for o in
                   Scenario(np.random.default_rng(1), 3, pedestrians=False).obstacles)


def test_pedestrians_never_start_on_the_drone():
    for seed in range(40):
        s = Scenario(np.random.default_rng(seed), 3)
        for o in s.obstacles:
            if o.kind == "pedestrian":
                gap = np.linalg.norm(s.drone_start - o.pos) - o.radius - C.DRONE_RADIUS
                assert gap > 0.5


def test_key_char_handles_special_and_unknown_keys():
    from drone_follow.live import key_char
    assert key_char(ord("o")) == "o" and key_char(ord("3")) == "3"
    for weird in (-1, 0, 27, 65296, 2**40, -2**31):          # arrows, shift, unknown codes
        assert key_char(weird) == ""


def test_hazard_terms_rise_for_a_hazard_on_a_collision_course():
    env = make_env(difficulty=0, seed=6)()
    env.reset(seed=6)
    sim = env.unwrapped
    env.step(np.zeros(3))
    assert sim._hazard_terms() == (0.0, 0.0)                       # empty sky
    ahead = sim.pos[0] + [1.0, 0.0, 0.0]
    sim.scenario.obstacles.append(Obstacle(ahead.copy(), np.array([-2.0, 0, 0]), 0.2, "rival"))
    approaching = sim._hazard_terms()
    sim.scenario.obstacles[-1].vel = np.array([2.0, 0, 0])          # same place, flying away
    receding = sim._hazard_terms()
    assert approaching[0] > 0.5 and receding[0] == 0.0 and approaching[1] > 0.0
    env.close()
