import json

from drone_follow.pick_winner import load, score


def _write(tmp, name, coll, err, accel, rate="0", tilt="0"):
    d = tmp / name
    d.mkdir()
    res = {f"ppo/level{l}": {"collision_rate": coll, "mean_follow_error_m": err, "p95_accel_ms2": accel,
                             "collision_kinds": {}} for l in (1, 2, 3)}
    (d / "final_eval_best.json").write_text(json.dumps(res))
    (d / "env.txt").write_text(f"DF_ACTION_RATE_LIMIT={rate} DF_R_TILT={tilt}")


def test_score_prefers_fewer_collisions_then_smoother(tmp_path):
    _write(tmp_path, "a", 0.10, 0.3, 8.0)
    _write(tmp_path, "b", 0.02, 0.3, 8.0)
    _write(tmp_path, "c", 0.02, 0.3, 3.0)
    s = {n: score(load(str(tmp_path / n))[0]) for n in "abc"}
    assert s["c"] < s["b"] < s["a"]


def test_load_reads_experiment_switches_and_skips_unfinished(tmp_path):
    _write(tmp_path, "x", 0.05, 0.3, 5.0, rate="0.1", tilt="0.5")
    (tmp_path / "unfinished").mkdir()
    res, env = load(str(tmp_path / "x"))
    assert env["DF_ACTION_RATE_LIMIT"] == "0.1" and env["DF_R_TILT"] == "0.5"
    assert load(str(tmp_path / "unfinished")) is None
