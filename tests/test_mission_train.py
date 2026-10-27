import json

from drone_follow import mission_train


class FakeModel:
    class logger:
        @staticmethod
        def record(*a, **k):
            pass

    def __init__(self):
        self.saved = []

    def predict(self, obs, deterministic=True):
        return obs[:3] * 0, None

    def save(self, path):
        self.saved.append(path)


def test_eval_keeps_the_best_score_and_logs_every_eval(tmp_path, monkeypatch):
    state = {}

    def fake_run_legs(policy, level, legs):
        crash, success = state["now"]
        return {"crash_rate": crash, "success_rate": success}

    monkeypatch.setattr(mission_train, "run_legs", fake_run_legs)
    cb = mission_train.MissionEval(str(tmp_path), every=1, legs=3, levels=(1, 2))
    cb.model = FakeModel()
    for outcome in [(0.5, 0.4), (0.2, 0.7), (0.3, 0.6)]:         # (crash, success) per eval, all levels
        state["now"] = outcome
        cb.run_eval()
    best = json.loads((tmp_path / "best.json").read_text())
    log = [json.loads(l) for l in (tmp_path / "eval_log.jsonl").read_text().splitlines()]
    assert best["L1_crash"] == 0.2 and len(log) == 3 and [e["best"] for e in log] == [True, True, False]
    assert sum(p.endswith("best.zip") for p in cb.model.saved) == 2
