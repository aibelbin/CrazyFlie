"""Train a PPO policy with a curriculum: clear sky -> static debris -> moving hazards -> stress.

    python -m drone_follow.train --out runs/ppo
"""
import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from drone_follow.env import make_env

# (difficulty, timesteps) per curriculum stage
STAGES = [(0, 200_000), (1, 500_000), (2, 800_000), (3, 800_000)]


class CollisionRate(BaseCallback):
    """Logs the fraction of finished episodes that ended in a collision."""

    def __init__(self):
        super().__init__()
        self._done = self._crashed = 0

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self._done += 1
                self._crashed += int(info.get("collision", False))
        return True

    def _on_rollout_end(self):
        if self._done:
            self.logger.record("custom/collision_rate", self._crashed / self._done)
            self._done = self._crashed = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all stage lengths")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="continue from this PPO .zip")
    ap.add_argument("--stages", default=None,
                    help="override curriculum, e.g. '2:600000,3:900000' (difficulty:steps)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mix", default=None,
                    help="comma list of difficulties assigned round-robin to the parallel envs, "
                         "e.g. '2,3' (use with --steps)")
    ap.add_argument("--steps", type=int, default=0, help="total timesteps in --mix mode")
    args = ap.parse_args()
    stages = STAGES if not args.stages else [
        (int(d), int(n)) for d, n in (x.split(":") for x in args.stages.split(","))]
    os.makedirs(args.out, exist_ok=True)

    venv = VecMonitor(SubprocVecEnv(
        [make_env(difficulty=0, seed=args.seed * 1000 + i) for i in range(args.envs)]))
    tb = os.path.join(args.out, "tb")
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu", tensorboard_log=tb,
                         custom_objects={"learning_rate": args.lr, "n_steps": 512, "batch_size": 512})
    else:
        model = PPO(
            "MlpPolicy", venv, n_steps=512, batch_size=512, learning_rate=args.lr, gamma=0.99,
            gae_lambda=0.95, ent_coef=0.0, seed=args.seed, device="cpu",
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            tensorboard_log=tb, verbose=1)

    ckpt = CheckpointCallback(save_freq=max(1, 250_000 // args.envs), save_path=args.out,
                              name_prefix="ckpt")
    if args.mix:
        mix = [int(x) for x in args.mix.split(",")]
        for i in range(args.envs):
            venv.env_method("set_difficulty", mix[i % len(mix)], indices=[i])
        print(f"=== mixed difficulties {mix}, {args.steps} steps ===", flush=True)
        model.learn(args.steps, reset_num_timesteps=False,
                    callback=[CollisionRate(), ckpt], tb_log_name="mixed")
        model.save(os.path.join(args.out, "final"))
        venv.close()
        return

    for difficulty, steps in stages:
        venv.env_method("set_difficulty", difficulty)
        print(f"=== stage: difficulty {difficulty}, {int(steps * args.scale)} steps ===", flush=True)
        model.learn(int(steps * args.scale), reset_num_timesteps=False,
                    callback=[CollisionRate(), ckpt], tb_log_name=f"stage{difficulty}")
        model.save(os.path.join(args.out, f"stage{difficulty}"))
    model.save(os.path.join(args.out, "final"))
    venv.close()


if __name__ == "__main__":
    main()
