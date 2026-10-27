import numpy as np
import pytest

from drone_follow import config as C
from drone_follow.mission_env import M_OBS_DIM, M_RANGES
from drone_follow.mission_policies import (CONE_ELEVS, CRUISE, LOW_ELEV, SIN_LOW, BendyRuler, Supervisor,
                                           potential_field, straight_line)

ELEVS = np.array(C.MISSION_ELEVS_DEG)
N_AZ = C.MISSION_N_AZ
AZ = (np.arange(N_AZ) * 360.0 / N_AZ + 180) % 360 - 180          # 0 = toward the waypoint


def make_obs(dist=50.0, blocked=(), agl=0.0, wp_up=0.0):
    """`blocked` = [(max |azimuth| deg, set of elevations, range m)] applied to an all-clear strip; `agl` = height
    above the ground (m), `wp_up` = height of the waypoint above the drone (m)."""
    obs = np.zeros(M_OBS_DIM, dtype=np.float32)
    obs[0] = dist / 100.0
    obs[1] = wp_up / 10.0
    obs[7] = agl / 10.0
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
    obs = make_obs(blocked=[(90, {-15.0, 0.0}, 6.0)])         # 15 deg up is clear, sideways is blocked to 90 deg
    a = BendyRuler()(obs)
    assert a[2] > 0 and a[0] > 0 and np.linalg.norm(a) == pytest.approx(CRUISE)


def test_bendy_ruler_climbs_steeper_when_the_ring_below_a_direction_is_close():
    # the level ring reads clear, but the ring below it sees a canopy 2 m away: flying level would skim it
    obs = make_obs(blocked=[(90, {-35.0, -15.0}, 2.0)])
    a = BendyRuler()(obs)
    assert a[2] > 0 and a[0] > 0                                # level is blocked by the sector below: climb
    assert BendyRuler()(make_obs(blocked=[(90, {-35.0}, 2.0)])) == pytest.approx(straight_line(make_obs()))


def test_bendy_ruler_level_candidate_needs_the_sector_below_it_clear():
    # only the sector straight below the waypoint direction is short (12 m < 13 m): the level direction is blocked
    obs = make_obs(blocked=[(5, {-15.0}, 12.0)])
    a = BendyRuler()(obs)
    assert not a == pytest.approx(straight_line(obs))
    assert BendyRuler()(make_obs(blocked=[(5, {-15.0}, 13.5)])) == pytest.approx(straight_line(make_obs()))


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
    sup(make_obs(blocked=[(45, {-35.0, -15.0}, 6.0)], agl=6.0 * SIN_LOW))   # the ground ahead and below
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
    ground = make_obs(blocked=[(45, {-15.0, -35.0}, 8.0)], agl=8.0 * SIN_LOW)   # only the ground is near
    for _ in range(3):
        sup(ground)
    assert not sup.guided


def test_supervisor_triggers_on_a_canopy_close_under_the_path_but_not_on_flat_ground():
    assert LOW_ELEV == -15.0
    sup = Supervisor(lambda obs: np.array([0.0, 0.7, 0.0]))
    sup.reset()
    flat = 5.0 / SIN_LOW                                               # ~19.3 m: flat ground 5 m below
    for _ in range(5):
        sup(make_obs(blocked=[(180, {-15.0}, flat)], agl=5.0))
    assert not sup.guided
    sup(make_obs(blocked=[(180, {-15.0}, 0.8 * flat)], agl=5.0))         # a bit short, but over 0.6 x flat
    assert not sup.guided
    sup(make_obs(blocked=[(180, {-15.0}, flat), (20, {-15.0}, 4.0)], agl=5.0))   # canopy top ahead and below
    assert sup.guided


def test_supervisor_low_ring_watches_only_the_cone_azimuths():
    sup = Supervisor(lambda obs: np.array([0.0, 0.7, 0.0]), cone_deg=45.0)
    sup.reset()
    sup(make_obs(blocked=[(180, {-15.0}, 4.0), (60, {-15.0}, 19.0)], agl=5.0))   # short only beyond 60 deg
    assert not sup.guided


def test_supervisor_hands_back_only_when_the_low_ring_is_clear_too():
    sup = Supervisor(lambda obs: np.array([0.0, 0.7, 0.0]), hold_s=0.1)
    sup.reset()
    sup(make_obs(blocked=[(20, {-15.0}, 4.0)], agl=5.0))
    assert sup.guided
    over = make_obs(blocked=[(20, {-15.0}, 12.5)], agl=5.0)          # over 0.6 x 19.3 m, under 0.7 x 19.3 m
    for _ in range(10):
        sup(over)
    assert sup.guided
    for _ in range(3):
        sup(make_obs(blocked=[(180, {-15.0}, 5.0 / SIN_LOW)], agl=5.0))
    assert not sup.guided


@pytest.mark.parametrize("blocked, agl", [
    ([(10, {-35.0}, 1.5)], 5.0),                     # a roof straight under the path
    ([(45, {-35.0}, 2.9)], 5.0),                     # at the edge of the cone
    ([(10, {-15.0}, 2.5)], 0.5),                     # low over the ground, a bump ahead (too low to trigger)
])
def test_supervisor_auto_never_descends_onto_something_close_below(blocked, agl):
    sup = Supervisor(lambda obs: np.array([0.0, 0.7, -0.9]))
    sup.reset()
    obs = make_obs(blocked=blocked, agl=agl, wp_up=-5.0)              # the waypoint is 5 m below
    assert straight_line(obs)[2] < 0
    a = sup(obs)
    assert not sup.guided and a[2] >= 0.0 and a[0] > 0


def test_supervisor_auto_descends_when_nothing_is_close_below():
    sup = Supervisor(lambda obs: np.array([0.0, 0.7, 0.0]))
    sup.reset()
    obs = make_obs(blocked=[(180, {-35.0}, 5.0 / np.sin(np.radians(35.0)))], agl=5.0, wp_up=-5.0)
    assert sup(obs) == pytest.approx(straight_line(obs)) and sup(obs)[2] < 0
