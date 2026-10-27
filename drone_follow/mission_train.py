"""PPO on mission legs: the RL avoider. Meant for gpu-sys (never the laptop).

    python -m drone_follow.mission_train --out runs/mission_s1 --seed 1 --total-steps 5000000

The policy learns whole legs (it is not wrapped in the supervisor during training); mission_evaluate scores
it both alone (`rl_alone`) and under the supervisor (`rl`). Every --eval-every steps it scores itself on fixed
legs and keeps best.zip by crash rate, then success rate.
"""
import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from drone_follow.mission_env import make_mission_env
from drone_follow.mission_evaluate import run_legs
from drone_follow.train import CollisionRate
from drone_follow.train_long import LatestSaver, ProgressWriter, TimeBudget

MISSION_MIX = [0, 1, 1, 2, 2, 3, 3]          # levels cycled over the parallel envs
EVAL_LEVELS = (1, 2, 3)


class MissionEval(BaseCallback):
    """Scores the policy alone on fixed legs; keeps the safest checkpoint (score: lower is better)."""

    def __init__(self, out: str, every: int, legs: int, levels=EVAL_LEVELS):
        super().__init__()
        self.out, self.every, self.legs, self.levels = out, every, legs, levels
        self.last, self.best = 0, np.inf
        try:                                    # resuming: keep the best score from before
            with open(os.path.join(out, "best.json")) as f:
                self.best = float(json.load(f)["score"])
        except (OSError, ValueError, KeyError):
            pass

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self.run_eval()
        return True

    def run_eval(self):
        policy = lambda obs: self.model.predict(obs, deterministic=True)[0]
        metrics = {}
        for lvl in self.levels:
            r = run_legs(policy, lvl, self.legs)          # builds and closes its own env: no leak across evals
            metrics[f"L{lvl}_crash"] = r["crash_rate"]
            metrics[f"L{lvl}_success"] = r["success_rate"]
        score = float(np.mean([metrics[f"L{l}_crash"] for l in self.levels])
                      + 0.5 * np.mean([1.0 - metrics[f"L{l}_success"] for l in self.levels]))
        for k, v in metrics.items():
            self.logger.record(f"eval/{k}", v)
        self.model.save(os.path.join(self.out, "latest.zip"))
        improved = bool(score < self.best)
        if improved:
            self.best = score
            self.model.save(os.path.join(self.out, "best.zip"))
            with open(os.path.join(self.out, "best.json"), "w") as f:
                json.dump({"steps": self.num_timesteps, "score": score, **metrics}, f, indent=2)
        with open(os.path.join(self.out, "eval_log.jsonl"), "a") as f:
            f.write(json.dumps({"steps": self.num_timesteps, "score": score, "best": improved, **metrics}) + "\n")
        print(f"[{self.num_timesteps}] eval {metrics} score={score:.3f}{' *best*' if improved else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--envs", type=int, default=7)
    ap.add_argument("--net", type=int, default=256)
    ap.add_argument("--total-steps", type=int, default=5_000_000)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--eval-every", type=int, default=1_000_000)
    ap.add_argument("--eval-legs", type=int, default=20)
    ap.add_argument("--lr-start", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=2e-5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)

    def lr_schedule(progress_remaining: float) -> float:
        return args.lr_end + (args.lr_start - args.lr_end) * progress_remaining

    venv = VecMonitor(SubprocVecEnv([make_mission_env(level=MISSION_MIX[i % len(MISSION_MIX)],
                                                      seed=args.seed * 1000 + i) for i in range(args.envs)]))
    n_steps = 2048
    batch = min(2048, n_steps * args.envs // 4)
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu", custom_objects={"learning_rate": lr_schedule})
    else:
        model = PPO(
            "MlpPolicy", venv, n_steps=n_steps, batch_size=batch, n_epochs=10,
            learning_rate=lr_schedule, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, max_grad_norm=0.5, seed=args.seed, device="cpu",
            policy_kwargs=dict(net_arch=dict(pi=[args.net, args.net], vf=[args.net, args.net]), log_std_init=-1.0),
            tensorboard_log=os.path.join(args.out, "tb"), verbose=1)
    print(f"config: {vars(args)}  rollout={n_steps * args.envs} batch={batch}", flush=True)

    evaluator = MissionEval(args.out, args.eval_every, args.eval_legs)
    callbacks = [CollisionRate(), evaluator, LatestSaver(args.out),
                 ProgressWriter(args.out, args.total_steps, args.max_hours), TimeBudget(args.max_hours)]
    model.learn(max(args.total_steps - model.num_timesteps, 1), reset_num_timesteps=False,
                callback=callbacks, tb_log_name="mission")
    model.save(os.path.join(args.out, "final"))
    evaluator.run_eval()
    venv.close()
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
