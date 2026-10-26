import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

from drone_follow.env import make_env
from drone_follow.web import Sim, make_handler


def _spectator_env(tmp_path, every=5, difficulty=2):
    live = str(tmp_path / "state.json")
    spec = dict(path=live, label="test", every=every)
    return make_env(difficulty=difficulty, seed=1, spectate=spec)(), live


def test_spectator_writes_poses_only_while_watched(tmp_path):
    env, live = _spectator_env(tmp_path)
    env.reset(seed=1)
    for _ in range(90):                                   # nobody watching: nothing is written
        env.step(np.zeros(3))
    assert not os.path.exists(live)
    (tmp_path / "watch").write_text("now")               # a viewer appears
    for _ in range(90):
        env.step(np.zeros(3))
    msg = json.loads(open(live).read())
    assert len(msg["drone"]["p"]) == 3 and msg["status"] and "test" in msg["policy"]
    assert json.loads(open(tmp_path / "world.json").read())["ep"] == msg["ep"]
    env.close()


def test_spectator_survives_episode_resets_and_writes_terrain(tmp_path):
    env, live = _spectator_env(tmp_path, every=3, difficulty=4)
    (tmp_path / "watch").write_text("now")
    for ep in range(3):
        env.reset(seed=ep)
        for _ in range(40):
            env.step(np.array([1.0, 0.0, 0.0]))
    assert os.path.exists(live)
    world = json.loads(open(tmp_path / "world.json").read())
    assert world["ep"] == 3 and world["terrain"]["n"] == 121 and len(world["statics"]) > 40
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


def test_viewer_page_static_files_and_event_stream(tmp_path):
    sim = Sim({}, "reactive", 2, 0)
    handler = make_handler(sim, str(tmp_path), [])
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/play").read().decode()
        assert "/static/viewer.js" in page and "<canvas" in page
        for name in ("three.min.js", "OrbitControls.js", "viewer.js"):
            assert len(urllib.request.urlopen(f"{base}/static/{name}").read()) > 1000
        for bad in ("/static/web.py", "/static/..%2Fweb.py", "/static/nope.js"):
            try:
                urllib.request.urlopen(base + bad)
                raise AssertionError(f"{bad} should 404")
            except urllib.error.HTTPError as e:
                assert e.code == 404
        with sim.cond:                                    # publish one snapshot, then read it as an event
            sim.state, sim.frame_no = json.dumps({"ep": 1, "status": "FOLLOW TARGET"}), 1
        stream = urllib.request.urlopen(base + "/events", timeout=5)
        line = stream.readline().decode()
        assert line.startswith("data: ") and json.loads(line[6:])["ep"] == 1
        stream.close()
    finally:
        server.shutdown()


def test_publish_snapshot_has_everything_the_viewer_draws():
    env = make_env(difficulty=3, seed=2)()
    env.reset(seed=2)
    for _ in range(5):
        env.step(np.zeros(3))
    sim = Sim({}, "reactive", 3, 2)
    sim.publish(env.unwrapped)
    msg = json.loads(sim.state)
    assert len(msg["drone"]["p"]) == 3 and len(msg["drone"]["q"]) == 4 and len(msg["person"]["p"]) == 2
    assert {o["k"] for o in msg["obs"]} >= {"debris", "pedestrian"}
    assert all(set(o) >= {"id", "p", "v", "r", "a", "g"} for o in msg["obs"])
    env.close()


def test_training_worker_view_streams_poses_and_world(tmp_path):
    run = tmp_path / "main_s1"
    run.mkdir()
    (run / "state.json").write_text(json.dumps({"ep": 2, "status": "FOLLOW TARGET"}))
    (run / "world.json").write_text(json.dumps({"ep": 2, "terrain": None, "statics": []}))
    (run / "progress.json").write_text(json.dumps({"steps": 5, "total": 10, "pct": 50.0}))
    handler = make_handler(Sim({}, "reactive", 2, 0), str(tmp_path), [])           # runs found automatically
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert [r["id"] for r in json.load(urllib.request.urlopen(base + "/api/runs"))] == ["main_s1"]
        assert "viewer.js" in urllib.request.urlopen(base + "/watch?run=main_s1").read().decode()
        assert json.load(urllib.request.urlopen(base + "/train/main_s1/world"))["ep"] == 2
        stream = urllib.request.urlopen(base + "/train/main_s1/events", timeout=5)
        line = stream.readline().decode()
        assert line.startswith("data: ") and json.loads(line[6:])["ep"] == 2
        stream.close()
        assert (run / "watch").exists()                                             # asked the worker to write poses
    finally:
        server.shutdown()
