import numpy as np
import pybullet as p
import pytest

from drone_follow import config as C
from drone_follow.mission_env import (M_N_RAYS, M_OBS_DIM, M_PREV_ACTION, M_RANGES, make_mission_env,
                                      sector_index, sector_ranges)
from drone_follow.sensing import cast

N_AZ = C.MISSION_N_AZ
E0 = C.MISSION_ELEVS_DEG.index(0.0)
FWD = E0 * N_AZ                                # elevation 0, azimuth 0: straight at the waypoint
FWD_DOWN = (E0 - 1) * N_AZ                     # elevation -15, azimuth 0


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
    assert term and not trunc and info["failure"] == "out_of_bounds" and r < 0
    env.close()


def test_sector_binning():
    v = np.array([[10, 0, 0], [10, 0, -1.5], [10, 0, -5.0], [10, 0, 1.5], [10, 0, 5.0], [10, 0, 20.0],
                  [0, 10, 0], [10, 3.9, 0], [10, 4.3, 0], [10, -0.2, 0]], float)
    # 10 m ahead at -8.5 deg is past the -7.5 edge: the -15 bin; 63 deg up is the top bin; 21.3 deg left stays
    # in the 22.5 deg sector (edge at 11.25 deg on either side); -1.1 deg is the forward sector.
    assert sector_index(v).tolist() == [FWD, FWD_DOWN, (E0 - 2) * N_AZ, (E0 + 1) * N_AZ, (E0 + 2) * N_AZ,
                                        (E0 + 2) * N_AZ, FWD + N_AZ // 4, FWD + 1, FWD + 1, FWD]
    r = sector_ranges(np.array([[5.0, 0.0, 0.0], [3.0, 0.1, 0.0], [30.0, 0.0, 0.0]]))
    assert r[FWD] == pytest.approx(np.hypot(3.0, 0.1)) and (np.delete(r, FWD) == C.MISSION_RANGE).all()


def _add_obstacle(sim, sphere=None, cylinder=None):
    """Put an obstacle in the leg (so the analytic sensor knows it) and in PyBullet (so the rays can hit it)."""
    leg = sim.leg
    if sphere is not None:
        x, y, z, r = sphere
        leg.spheres = np.vstack([leg.spheres, [sphere]])
        leg.sphere_kind.append("rock")
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=sim.CLIENT)
        p.createMultiBody(0, col, basePosition=[x, y, z], physicsClientId=sim.CLIENT)
    if cylinder is not None:
        x, y, z0, z1, r = cylinder
        leg.cylinders = np.vstack([leg.cylinders, [cylinder]])
        leg.cylinder_kind.append("pole")
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=z1 - z0, physicsClientId=sim.CLIENT)
        p.createMultiBody(0, col, basePosition=[x, y, (z0 + z1) / 2], physicsClientId=sim.CLIENT)


def test_sectors_see_a_thin_pole_between_the_rays():
    env = make_mission_env(level=0, seed=7, length=60.0)()
    env.reset(seed=7)
    sim = env.unwrapped
    pos = sim.pos[0].copy()
    px, py = pos[0] + 15.0, pos[1] + 0.34                  # forward ray passes 0.19 m from the 0.15 m pole
    g = sim.leg.ground([px, py])
    _add_obstacle(sim, cylinder=[px, py, g - 0.5, g + 12.0, 0.15])
    ray = cast(sim.CLIENT, pos, np.array([[1.0, 0.0, 0.0]]), C.MISSION_RANGE, C.MISSION_RAY_START)
    assert ray[0] == pytest.approx(C.MISSION_RANGE)        # the forward ray alone misses it
    for _ in range(6):                                     # let the delayed, smoothed sensor settle
        obs, *_ = env.step(np.zeros(3))
    d = np.hypot(px - sim.pos[0][0], py - sim.pos[0][1]) - 0.15
    assert obs[M_RANGES][FWD] * C.MISSION_RANGE == pytest.approx(d, abs=0.6)
    env.close()


def test_a_sphere_ahead_and_below_lands_in_the_lower_sector():
    env = make_mission_env(level=0, seed=8, length=60.0)()
    env.reset(seed=8)
    sim = env.unwrapped
    pos = sim.pos[0].copy()
    _add_obstacle(sim, sphere=[pos[0] + 10.0, pos[1], pos[2] - 1.5, 0.5])
    for _ in range(6):
        obs, *_ = env.step(np.zeros(3))
    r = obs[M_RANGES] * C.MISSION_RANGE
    assert r[FWD_DOWN] == pytest.approx(np.hypot(10.0, 1.5) - 0.5, abs=0.6)
    assert r[FWD] > 15.0
    env.close()


def test_commands_reach_the_autopilot_one_step_late_and_obs_shows_the_latest():
    assert C.MISSION_ACT_DELAY == 1
    env = make_mission_env(level=0, seed=9, length=60.0)()
    env.reset(seed=9)
    sim = env.unwrapped
    speeds = []
    for k in (1, 2, 3):
        a = np.array([0.1 * k, 0.0, 0.0])
        obs, *_ = env.step(a)
        speeds.append(float(np.linalg.norm(sim._v_target)))
        assert obs[M_PREV_ACTION] == pytest.approx(a)        # the command just issued, not the one before
    assert speeds == pytest.approx([0.0, 0.1 * C.MISSION_SPEED_LIMIT, 0.2 * C.MISSION_SPEED_LIMIT])
    assert sim._v_target[0] > 0
    env.close()


def test_hovering_in_place_costs_reward():
    env = make_mission_env(level=0, seed=10, length=60.0)()
    env.reset(seed=10)
    rewards = []
    for _ in range(30):
        _, r, term, trunc, _ = env.step(np.zeros(3))
        assert not (term or trunc)
        rewards.append(r)
    assert sum(rewards) < 0 and np.mean(rewards) == pytest.approx(-C.MR_TIME, abs=0.003)
    env.close()
