"""Lets one training worker show what it is doing: writes a JPEG of its current episode.

The worker only renders while someone is watching: the web server touches `<dir>/watch`
every couple of seconds, and frames are written only if that file is fresh. When nobody is
watching this costs nothing.
"""
import json
import os
import time

import cv2
import gymnasium as gym

from drone_follow.drone_visual import DroneVisual
from drone_follow.render import render_frame, status_from

WATCH_TTL_S = 8.0
FRAME_W, FRAME_H = 640, 360


class SpectatorWrapper(gym.Wrapper):
    def __init__(self, env, path: str, label: str, every: int = 45, progress_path: str | None = None):
        super().__init__(env)
        self.path, self.label, self.every = path, label, every
        self.progress_path = progress_path
        self.flag = os.path.join(os.path.dirname(path), "watch")
        self.n, self.watching, self.visual = 0, False, None
        self.episodes_seen = 0

    def set_difficulty(self, difficulty: int):
        self.env.set_difficulty(difficulty)

    def reset(self, **kw):
        self.visual = None                       # the old bodies are gone after a reset
        self.episodes_seen += 1
        return self.env.reset(**kw)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.n += 1
        if self.n % 30 == 0:
            try:
                self.watching = time.time() - os.path.getmtime(self.flag) < WATCH_TTL_S
            except OSError:
                self.watching = False
        if self.watching and self.n % self.every == 0:
            self._write(info, term)
        return obs, r, term, trunc, info

    def _steps_label(self) -> str:
        if not self.progress_path:
            return ""
        try:
            with open(self.progress_path) as f:
                return f"  step {json.load(f)['steps'] / 1e6:.1f}M"
        except (OSError, ValueError, KeyError):
            return ""

    def _write(self, info: dict, term: bool):
        sim = self.env.unwrapped
        if self.visual is None:
            self.visual = DroneVisual(sim.CLIENT, int(sim.DRONE_IDS[0]))
        self.visual.update()
        title = f"{self.label}{self._steps_label()}  level {sim.difficulty}  (exploring, sped up)"
        img = render_frame(sim, FRAME_W, FRAME_H, status=status_from(info, term), title=title,
                           cam=(225.0, 27.0, 7.5))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp, self.path)
