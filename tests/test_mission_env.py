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
