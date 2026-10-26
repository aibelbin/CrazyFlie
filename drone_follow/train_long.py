"""Long PPO run with a stronger recipe. Meant for a remote machine (never the laptop).

    python -m drone_follow.train_long --out runs/long_a --seed 1 --net 256 \
        --total-steps 60000000 --max-hours 20

What differs from train.py:
  * trained from scratch on a permanent mix of difficulty levels (no forgetting easy cases)
  * shaped reward: time-to-collision and clearance terms, stronger smoothness penalty
  * bigger rollouts/batches, gamma 0.995, linearly decaying learning rate, low initial action noise
  * every --eval-every steps it scores the policy on fixed unseen scenarios, keeps best.zip
    (lowest collision rate), writes latest.zip, and logs to eval_log.jsonl
  * stops cleanly after --max-hours and can resume with --resume <out>/latest.zip
"""
import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from drone_follow import config as C
from drone_follow.env import make_env
from drone_follow.train import CollisionRate

WARMUP_STEPS = 3_000_000
WARMUP_MIX = [0, 1, 2, 2, 3, 3, 3]       # difficulty levels cycled over the parallel envs
MAIN_MIX = [1, 2, 2, 3, 3, 3, 3]
F450_WARMUP_MIX = [0, 1, 2, 2, 3, 4, 4]          # DF_DRONE=f450: hills, forest and wind enter early
F450_MAIN_MIX = [1, 2, 3, 4, 4, 5, 5]
EVAL_SEED0 = 20_000                       # distinct from evaluate.py's seeds (10_000+)


class PhaseMixer(BaseCallback):
    """Assigns a difficulty level to every parallel env, switching mix after the warm-up."""

    def __init__(self, n_envs: int):
        super().__init__()
        self.n_envs, self.phase = n_envs, None
        self.warmup_mix = F450_WARMUP_MIX if C.F450 else WARMUP_MIX
        self.main_mix = F450_MAIN_MIX if C.F450 else MAIN_MIX

    def _apply(self, mix):
        for i in range(self.n_envs):
            self.training_env.env_method("set_difficulty", mix[i % len(mix)], indices=[i])

    def _on_training_start(self):
        self._check()

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            self._check()
        return True

    def _check(self):
        phase = "warmup" if self.num_timesteps < WARMUP_STEPS else "main"
        if phase != self.phase:
            self.phase = phase
            self._apply(self.warmup_mix if phase == "warmup" else self.main_mix)
            print(f"[{self.num_timesteps}] difficulty mix -> {phase}", flush=True)


class BestByCollisions(BaseCallback):
    """Periodically scores the policy on fixed scenarios and keeps the safest checkpoint."""

    def __init__(self, out: str, every: int, episodes: int, levels=None):
        super().__init__()
        levels = levels or ((3, 4, 5) if C.F450 else (2, 3))        # the hardest levels of the setup being trained
        self.out, self.every, self.episodes, self.levels = out, every, episodes, levels
        self.envs, self.last, self.best = {}, 0, np.inf
        try:                                    # resuming: keep the best score from before
            with open(os.path.join(out, "best.json")) as f:
                self.best = float(json.load(f)["score"])
            print(f"restored previous best score {self.best:.4f}", flush=True)
        except (OSError, ValueError, KeyError):
            pass

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self.run_eval()
        return True

    def run_eval(self):
        metrics = {}
        for lvl in self.levels:
            env = self.envs.setdefault(lvl, make_env(difficulty=lvl, seed=EVAL_SEED0)())
            crashes, errs = 0, []
            for ep in range(self.episodes):
                obs, _ = env.reset(seed=EVAL_SEED0 + ep)
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = env.step(action)
                    errs.append(abs(info["follow_dist"] - C.FOLLOW_DIST))
                    done = term or trunc
                crashes += int(info["collision"])
            metrics[f"L{lvl}_collision"] = crashes / self.episodes
            metrics[f"L{lvl}_follow_err"] = float(np.mean(errs))
        score = float(np.mean([metrics[f"L{l}_collision"] for l in self.levels])
                      + 0.05 * np.mean([metrics[f"L{l}_follow_err"] for l in self.levels]))
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
            f.write(json.dumps({"steps": self.num_timesteps, "score": score, "best": improved,
                                **metrics}) + "\n")
        print(f"[{self.num_timesteps}] eval {metrics} score={score:.3f}{' *best*' if improved else ''}",
              flush=True)


class LatestSaver(BaseCallback):
    """Saves latest.zip often, so a viewer or a resume never loses much."""

    def __init__(self, out: str, every: int = 250_000):
        super().__init__()
        self.out, self.every, self.last = out, every, 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self.model.save(os.path.join(self.out, "latest.zip"))
        return True


class ProgressWriter(BaseCallback):
    """Writes progress.json every ~20 s for the web dashboard."""

    def __init__(self, out: str, total: int, max_hours: float, every_s: float = 20.0, window: int = 200):
        super().__init__()
        self.out, self.total, self.max_hours, self.every_s = out, total, max_hours, every_s
        self.recent = deque(maxlen=window)
        self.t_start = self.t_last = time.time()
        self.steps_last, self.sps = 0, 0.0

    def _on_training_start(self):
        self.t_start = self.t_last = time.time()
        self.steps_last = self.num_timesteps

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.recent.append(int(info.get("collision", False)))
        now = time.time()
        if now - self.t_last >= self.every_s:
            rate = (self.num_timesteps - self.steps_last) / (now - self.t_last)
            self.sps = rate if self.sps == 0 else 0.7 * self.sps + 0.3 * rate
            self.t_last, self.steps_last = now, self.num_timesteps
            self._write(now)
        return True

    def _write(self, now: float):
        buf = self.model.ep_info_buffer
        left = self.total - self.num_timesteps
        data = {
            "steps": int(self.num_timesteps), "total": int(self.total),
            "pct": 100.0 * self.num_timesteps / self.total,
            "sps": self.sps,
            "elapsed_h": (now - self.t_start) / 3600,
            "eta_h": left / self.sps / 3600 if self.sps > 0 else None,
            "budget_h": self.max_hours,
            "phase": "warm-up mix" if self.num_timesteps < WARMUP_STEPS else "main mix",
            "train_collision_rate": float(np.mean(self.recent)) if self.recent else None,
            "n_recent": len(self.recent),
            "ep_reward": float(np.mean([e["r"] for e in buf])) if len(buf) else None,
            "ep_len_s": float(np.mean([e["l"] for e in buf])) / C.CTRL_FREQ if len(buf) else None,
            "updated_at": now,
        }
        tmp = os.path.join(self.out, "progress.json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(self.out, "progress.json"))


class TimeBudget(BaseCallback):
    def __init__(self, hours: float):
        super().__init__()
        self.deadline = time.time() + hours * 3600

    def _on_step(self) -> bool:
        if time.time() > self.deadline:
            print(f"[{self.num_timesteps}] time budget reached, stopping", flush=True)
            return False
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--envs", type=int, default=7)
    ap.add_argument("--net", type=int, default=256, help="hidden width of the two-layer actor/critic")
    ap.add_argument("--total-steps", type=int, default=60_000_000)
    ap.add_argument("--max-hours", type=float, default=20.0)
    ap.add_argument("--eval-every", type=int, default=1_000_000)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--lr-start", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=2e-5)
    ap.add_argument("--threads", type=int, default=3, help="torch threads for the learner")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)

    def lr_schedule(progress_remaining: float) -> float:
        return args.lr_end + (args.lr_start - args.lr_end) * progress_remaining

    spectate = dict(path=os.path.join(args.out, "live.jpg"),
                    label=f"{os.path.basename(os.path.normpath(args.out))} worker",
                    progress_path=os.path.join(args.out, "progress.json"))
    venv = VecMonitor(SubprocVecEnv(
        [make_env(difficulty=2, seed=args.seed * 1000 + i,
                  spectate=spectate if i == 2 else None)     # worker 2 is level 2 in both mixes
         for i in range(args.envs)]))
    n_steps = 2048
    batch = min(2048, n_steps * args.envs // 4)
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu",
                         custom_objects={"learning_rate": lr_schedule})
    else:
        model = PPO(
            "MlpPolicy", venv, n_steps=n_steps, batch_size=batch, n_epochs=10,
            learning_rate=lr_schedule, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, max_grad_norm=0.5, seed=args.seed, device="cpu",
            policy_kwargs=dict(net_arch=dict(pi=[args.net, args.net], vf=[args.net, args.net]),
                               log_std_init=-1.0),
            tensorboard_log=os.path.join(args.out, "tb"), verbose=1)
    print(f"config: {vars(args)}  rollout={n_steps * args.envs} batch={batch}", flush=True)

    remaining = max(args.total_steps - model.num_timesteps, 1)
    callbacks = [
        CollisionRate(), PhaseMixer(args.envs),
        BestByCollisions(args.out, args.eval_every, args.eval_episodes),
        LatestSaver(args.out), ProgressWriter(args.out, args.total_steps, args.max_hours),
        CheckpointCallback(save_freq=max(1, 5_000_000 // args.envs), save_path=args.out,
                           name_prefix="ckpt"),
        TimeBudget(args.max_hours),
    ]
    model.learn(remaining, reset_num_timesteps=False, callback=callbacks, tb_log_name="long")
    model.save(os.path.join(args.out, "final"))
    callbacks[2].run_eval()                  # score the final policy too (index of BestByCollisions)
    venv.close()
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
