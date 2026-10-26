"""Score policies on fixed random scenarios: collision rate, following error, survival.

    python -m drone_follow.evaluate --episodes 50 --model runs/ppo/final.zip
"""
import argparse
import json

import numpy as np

from drone_follow import config as C
from drone_follow.env import make_env
from drone_follow.policies import naive_follow, reactive_follow


def run_episodes(policy, difficulty: int, episodes: int, seed0: int = 10_000,
                 pedestrians: bool = True) -> dict:
    env = make_env(difficulty=difficulty, seed=seed0, pedestrians=pedestrians)()
    crashes, errs, durations = 0, [], []
    action_change, accel = [], []                 # smoothness: command jitter, true acceleration
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done, steps, prev_a, prev_v = False, 0, None, None
        while not done:
            action = policy(obs)
            a = np.clip(np.asarray(action, float).reshape(3), -1.0, 1.0)
            obs, _, term, trunc, info = env.step(action)
            v = env.unwrapped.vel[0].copy()
            if prev_a is not None:
                action_change.append(float(np.linalg.norm(a - prev_a)))
                accel.append(float(np.linalg.norm(v - prev_v)) * C.CTRL_FREQ)
            prev_a, prev_v = a, v
            errs.append(abs(info["follow_dist"] - C.FOLLOW_DIST))
            steps += 1
            done = term or trunc
        crashes += int(info["collision"])
        durations.append(steps / C.CTRL_FREQ)
    env.close()
    return {"difficulty": difficulty, "episodes": episodes,
            "collision_rate": crashes / episodes,
            "mean_follow_error_m": float(np.mean(errs)),
            "mean_survival_s": float(np.mean(durations)),
            "mean_action_change": float(np.mean(action_change)),
            "mean_accel_ms2": float(np.mean(accel)),
            "p95_accel_ms2": float(np.percentile(accel, 95))}


def load_ppo_policy(path: str):
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--model", default=None, help="path to a trained PPO .zip")
    ap.add_argument("--difficulties", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--no-pedestrians", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true", help="only score --model")
    ap.add_argument("--out", default=None, help="write results to this JSON file")
    args = ap.parse_args()

    policies = {} if args.skip_baselines else {"naive_follow": naive_follow,
                                               "reactive_follow": reactive_follow}
    if args.model:
        policies["ppo"] = load_ppo_policy(args.model)

    results = {}
    print(f"{'policy':<16}{'level':>6}{'collisions':>12}{'follow err (m)':>16}{'survival (s)':>14}"
          f"{'cmd jitter':>12}{'accel p95':>11}")
    for name, pol in policies.items():
        for d in args.difficulties:
            r = run_episodes(pol, d, args.episodes, pedestrians=not args.no_pedestrians)
            results[f"{name}/level{d}"] = r
            print(f"{name:<16}{d:>6}{r['collision_rate']:>11.0%}"
                  f"{r['mean_follow_error_m']:>16.2f}{r['mean_survival_s']:>14.1f}"
                  f"{r['mean_action_change']:>12.3f}{r['p95_accel_ms2']:>11.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
