"""Lets one training worker show what it is doing: writes its current episode for the 3D viewer.

The worker only writes while someone is watching: the web server touches `<dir>/watch` every couple
of seconds, and files are written only if that flag is fresh. When nobody is watching this costs nothing.
Files (same format the /play viewer streams): `state.json` (poses, rewritten every few steps) and
`world.json` (terrain, trees and rocks, written once per episode).
"""
import os
import time

import gymnasium as gym

from drone_follow.render import status_from
from drone_follow.snapshot import pose_snapshot, world_snapshot

WATCH_TTL_S = 8.0


def _atomic_write(path: str, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


class SpectatorWrapper(gym.Wrapper):
    def __init__(self, env, path: str, label: str, every: int = 3, progress_path: str | None = None):
        """`path` is the run folder's `state.json`; `label` names the worker in the viewer."""
        super().__init__(env)
        self.path, self.label, self.every = path, label, every
        self.progress_path = progress_path
        self.world_path = os.path.join(os.path.dirname(path), "world.json")
        self.flag = os.path.join(os.path.dirname(path), "watch")
        self.n, self.watching = 0, False
        self.episodes_seen, self.crashes, self.world_ep = 0, 0, -1

    def set_difficulty(self, difficulty: int):
        self.env.set_difficulty(difficulty)

    def reset(self, **kw):
        self.episodes_seen += 1
        return self.env.reset(**kw)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.n += 1
        if info.get("collision") and (term or trunc):
            self.crashes += 1
        if self.n % 30 == 0:
            try:
                self.watching = time.time() - os.path.getmtime(self.flag) < WATCH_TTL_S
            except OSError:
                self.watching = False
        if self.watching and (self.n % self.every == 0 or term or trunc):
            self._write(info, term)
        return obs, r, term, trunc, info

    def _write(self, info: dict, term: bool):
        sim = self.env.unwrapped
        if self.world_ep != self.episodes_seen:
            _atomic_write(self.world_path, world_snapshot(sim.scenario, self.episodes_seen))
            self.world_ep = self.episodes_seen
        msg = pose_snapshot(sim, self.episodes_seen, status_from(info, term), f"{self.label} (exploring, sped up)",
                            sim.difficulty, self.episodes_seen - 1, self.crashes)
        _atomic_write(self.path, msg.encode())
