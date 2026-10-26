"""Record 3D clips of policies flying the same random scenario, side by side.

    python -m drone_follow.demo_sim --policies naive_follow ppo --model runs/ppo/final.zip \
        --difficulty 3 --seed 4 --out videos/compare.mp4
"""
import argparse

import cv2
import numpy as np

from drone_follow import config as C
from drone_follow.env import make_env
from drone_follow.evaluate import load_ppo_policy
from drone_follow.policies import naive_follow, reactive_follow
from drone_follow.render import render_frame, status_from


class Run:
    """One policy flying one scenario, advanced a frame at a time."""

    def __init__(self, name, policy, difficulty, seed):
        self.name, self.policy = name, policy
        self.env = make_env(difficulty=difficulty, seed=seed)()
        self.obs, _ = self.env.reset(seed=seed)
        self.done, self.crashed, self.status = False, False, "FOLLOW TARGET"

    def advance(self):
        if self.done:
            return
        self.obs, _, term, trunc, info = self.env.step(self.policy(self.obs))
        self.crashed = bool(info["collision"])
        self.status = status_from(info, term)
        self.done = term or trunc

    def frame(self, w, h):
        return render_frame(self.env.unwrapped, w, h, status=self.status, title=self.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["naive_follow", "reactive_follow"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", default="videos/compare.mp4")
    ap.add_argument("--size", type=int, nargs=2, default=[640, 480])
    ap.add_argument("--hold", type=float, default=1.5, help="seconds to hold the final frame")
    args = ap.parse_args()

    table = {"naive_follow": naive_follow, "reactive_follow": reactive_follow}
    if args.model:
        table["ppo"] = load_ppo_policy(args.model)
    runs = [Run(n, table[n], args.difficulty, args.seed) for n in args.policies]

    w, h = args.size
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), C.CTRL_FREQ, (w * len(runs), h))
    step = 0
    while not all(r.done for r in runs):
        for r in runs:
            r.advance()
        if step % 1 == 0:
            out.write(np.hstack([r.frame(w, h) for r in runs]))
        step += 1
    last = np.hstack([r.frame(w, h) for r in runs])
    for _ in range(int(args.hold * C.CTRL_FREQ)):
        out.write(last)
    out.release()
    print(f"wrote {args.out}: " + ", ".join(
        f"{r.name}={'CRASHED' if r.crashed else 'survived'}" for r in runs))


if __name__ == "__main__":
    main()
