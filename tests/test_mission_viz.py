import numpy as np

from drone_follow.mission_evaluate import make_policy
from drone_follow.mission_viz import animate, fly, plot


def test_fly_records_the_path_and_plots_it(tmp_path):
    leg, pos, guided, info = fly(make_policy("bendy"), level=1, seed=3, length=40.0)
    assert pos.shape[1] == 3 and len(pos) == len(guided) > 10
    assert np.allclose(pos[0], leg.start, atol=0.05)
    _, pos2, guided2, info2 = fly(make_policy("straight"), level=1, seed=3, length=40.0)
    flights = {"bendy": (pos, guided, info), "straight": (pos2, guided2, info2)}
    png, gif = tmp_path / "leg.png", tmp_path / "leg.gif"
    plot(leg, flights, str(png), "test")
    animate(leg, flights, str(gif), "test", every=30)
    assert png.stat().st_size > 10_000 and gif.stat().st_size > 10_000
