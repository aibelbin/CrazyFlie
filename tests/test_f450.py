"""The F450 model is selected by an environment variable read at import, so each check runs in a
fresh interpreter."""
import json
import os
import subprocess
import sys

SCRIPT = r"""
import json, numpy as np
from drone_follow import config as C
from drone_follow.env import FollowAvoidAviary, SingleDroneWrapper
env = SingleDroneWrapper(FollowAvoidAviary(gui=False, difficulty=0, seed=1, episode_sec=1e9))
sim = env.unwrapped
env.reset(seed=1)
for _ in range(60):
    env.step(np.zeros(3))
hover_z = float(sim.pos[0][2]); hover_speed = float(np.linalg.norm(sim.vel[0]))
for _ in range(90):
    env.step(np.array([1.0, 0, 0]))
v_fwd = float(sim.vel[0][0]); tilt = float(sim._tilt)
print(json.dumps(dict(f450=C.F450, hover_z=hover_z, hover_speed=hover_speed, v_fwd=v_fwd, tilt=tilt,
                      mass=float(sim.M), limit=C.SPEED_LIMIT)))
"""


def _run(**env_vars):
    env = dict(os.environ, PYTHONPATH=os.getcwd(), **env_vars)
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_f450_hovers_and_tracks_a_velocity_command():
    r = _run(DF_DRONE="f450")
    assert r["f450"] and abs(r["mass"] - 1.5) < 1e-6
    assert abs(r["hover_z"] - 1.2) < 0.05 and r["hover_speed"] < 0.05
    assert abs(r["v_fwd"] - r["limit"]) < 0.15            # reaches the commanded speed
    assert r["tilt"] < 0.7                                # stays well inside the flip limit


def test_default_drone_is_still_the_crazyflie():
    r = _run()
    assert not r["f450"] and r["mass"] < 0.05 and r["limit"] == 1.5
