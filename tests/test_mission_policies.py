import numpy as np
import pytest

from drone_follow import config as C
from drone_follow.mission_env import M_OBS_DIM, M_RANGES
from drone_follow.mission_policies import CONE_ELEVS, CRUISE, BendyRuler, Supervisor, potential_field, straight_line

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


def test_potential_field_never_pushes_down_or_up():
    obs = make_obs(blocked=[(30, {15.0, 35.0}, 2.0)])          # something close overhead-ahead
    assert potential_field(obs)[2] == pytest.approx(0.0)


def test_bendy_ruler_climbs_over_a_low_wide_wall():
    obs = make_obs(blocked=[(90, {-15.0, 0.0, 15.0}, 6.0)])   # 35 deg up is clear, sideways is blocked to 90 deg
    a = BendyRuler()(obs)
    assert a[2] > 0 and a[0] > 0 and np.linalg.norm(a) == pytest.approx(CRUISE)


def test_bendy_ruler_goes_around_a_tall_narrow_obstacle():
    obs = make_obs(blocked=[(25, {-15.0, 0.0, 15.0, 35.0}, 6.0)])
    a = BendyRuler()(obs)
    assert abs(a[1]) > 0.3 and a[0] > 0


def test_bendy_ruler_needs_lookahead_plus_margin_to_call_a_direction_open():
    obs = make_obs(blocked=[(25, set(ELEVS), 11.0)])           # forward reads 11 m: past lookahead, short of +margin
    a = BendyRuler()(obs)
    assert abs(a[1]) > 0.3 or a[2] > 0
    assert BendyRuler(margin=0.0)(obs) == pytest.approx(straight_line(obs))


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


def test_supervisor_cone_is_level_and_first_up_ring_only():
    assert list(CONE_ELEVS) == [0.0, 15.0]
    sup = Supervisor(lambda obs: np.array([0.0, 0.7, 0.0]))
    sup.reset()
    sup(make_obs(blocked=[(45, {-35.0, -15.0}, 6.0)]))                # the ground ahead and below: not a trigger
    assert not sup.guided
    sup(make_obs(blocked=[(60, {35.0}, 6.0)]))                         # steeply up: not a trigger
    assert not sup.guided
    sup(make_obs(blocked=[(45, {15.0}, 8.0)]))
    assert sup.guided


def test_supervisor_hands_back_only_when_the_whole_trigger_cone_is_clear():
    sup = Supervisor(lambda obs: np.array([0.0, 0.7, 0.0]), hold_s=0.1)
    sup.reset()
    sup(make_obs(blocked=[(30, {0.0}, 8.0)]))
    assert sup.guided
    edge = make_obs(blocked=[(45, {0.0}, 14.0)])                      # 45 deg off still under trigger + margin
    for _ in range(10):
        sup(edge)
    assert sup.guided
    ground = make_obs(blocked=[(45, {-15.0, -35.0}, 8.0)])            # only the ground is near: hand back
    for _ in range(3):
        sup(ground)
    assert not sup.guided
