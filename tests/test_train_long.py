import numpy as np

from drone_follow import config as C
from drone_follow import train_long


class FakeEnv:
    def reset(self, seed=None):
        return np.zeros(3), {}

    def step(self, action):
        return np.zeros(3), 0.0, True, False, {"follow_dist": C.FOLLOW_DIST, "collision": False}


class FakeModel:
    class logger:
        @staticmethod
        def record(*a, **k):
            pass

    def predict(self, obs, deterministic=True):
        return np.zeros(3), None

    def save(self, path):
        pass


def test_eval_builds_each_level_world_once(tmp_path, monkeypatch):
    """Each eval used to build a fresh world per level and never close it: ~140 MB leaked per eval."""
    builds = []

    def fake_make_env(difficulty, seed):
        builds.append(difficulty)
        return FakeEnv

    monkeypatch.setattr(train_long, "make_env", fake_make_env)
    cb = train_long.BestByCollisions(str(tmp_path), every=1, episodes=2, levels=(3, 5))
    cb.model = FakeModel()
    for _ in range(3):
        cb.run_eval()
    assert sorted(builds) == [3, 5]
