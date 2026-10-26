import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

from drone_follow.env import make_env
from drone_follow.web import Sim, make_handler


def _spectator_env(tmp_path, every=5):
    live = str(tmp_path / "live.jpg")
    spec = dict(path=live, label="test", every=every)
    return make_env(difficulty=2, seed=1, spectate=spec)(), live


def test_spectator_writes_frames_only_while_watched(tmp_path):
    env, live = _spectator_env(tmp_path)
    env.reset(seed=1)
    for _ in range(90):                                   # nobody watching: no rendering
        env.step(np.zeros(3))
    assert not os.path.exists(live)
    (tmp_path / "watch").write_text("now")               # a viewer appears
    for _ in range(90):
        env.step(np.zeros(3))
    assert os.path.getsize(live) > 1000
    env.close()


def test_spectator_survives_episode_resets(tmp_path):
    env, live = _spectator_env(tmp_path, every=3)
    (tmp_path / "watch").write_text("now")
    for ep in range(3):
        env.reset(seed=ep)
        for _ in range(40):
            env.step(np.array([1.0, 0.0, 0.0]))
    assert os.path.exists(live)
    env.close()


def test_dashboard_endpoints(tmp_path):
    run = tmp_path / "long_a"
    run.mkdir()
    (run / "progress.json").write_text(json.dumps({"steps": 5, "total": 10, "pct": 50.0}))
    (run / "eval_log.jsonl").write_text(
        json.dumps({"steps": 1, "L2_collision": 0.5, "L3_collision": 0.6}) + "\n" + "garbage\n")
    handler = make_handler(Sim({}, "reactive", 2, 0), str(tmp_path), [{"id": "long_a", "title": "Run A"}])
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "long_a" in page and "Drone follow-and-avoid" in page
        data = json.load(urllib.request.urlopen(base + "/train/long_a/progress"))
        assert data["progress"]["steps"] == 5 and len(data["evals"]) == 1 and data["best"] is None
        for bad in ("/train/other/progress", "/train/../long_a/progress", "/nope"):
            try:
                urllib.request.urlopen(base + bad)
                raise AssertionError(f"{bad} should 404")
            except urllib.error.HTTPError as e:
                assert e.code == 404
    finally:
        server.shutdown()
