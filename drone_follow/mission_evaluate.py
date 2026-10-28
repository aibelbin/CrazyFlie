"""Score avoiders on fixed, seeded mission legs (the same legs for every policy).

    python -m drone_follow.mission_evaluate --policy bendy --level 2 --legs 500 --out results/mission/bendy_L2.json
    python -m drone_follow.mission_evaluate --table results/mission/*.json

Heavy (measured: ~5 min per 500 legs per policy and level): run on gpu-sys, never the laptop.
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

from drone_follow import config as C
from drone_follow.mission_env import make_mission_env
from drone_follow.mission_policies import BendyRuler, Supervisor, load_ppo, potential_field, straight_line

EVAL_SEED0 = 30_000                      # test legs. SB3's PPO(seed=s) reseeds the training envs as s + i (small
                                         # seeds, then each env's own generator), so training never replays these
                                         # legs; mission_train's checkpoint selection uses VAL_SEED0 = 40_000
POLICIES = ("straight", "potential", "bendy", "rl", "rl_alone")


def make_policy(name: str, model: str | None = None):
    if name == "straight":
        return straight_line
    if name == "potential":
        return Supervisor(potential_field)
    if name == "bendy":
        return Supervisor(BendyRuler())
    if name in ("rl", "rl_alone"):
        if not model:
            raise ValueError(f"policy {name} needs --model")
        ppo = load_ppo(model)
        return Supervisor(ppo) if name == "rl" else ppo
    raise ValueError(f"unknown policy {name}")


def run_legs(policy, level: int, legs: int, seed0: int = EVAL_SEED0, length: float | None = None) -> dict:
    env = make_mission_env(level=level, seed=seed0, length=length)()
    sim = env.unwrapped
    crashes = successes = 0
    extra, time_ratio, guided, action_change, accel = [], [], [], [], []
    kinds, failures = Counter(), Counter()
    for ep in range(legs):
        obs, _ = env.reset(seed=seed0 + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        done, steps, guided_steps, prev_a, prev_v = False, 0, 0, None, None
        while not done:
            a = np.clip(np.asarray(policy(obs), float).reshape(3), -1.0, 1.0)
            guided_steps += int(getattr(policy, "guided", False))
            obs, _, term, trunc, info = env.step(a)
            v = sim.vel[0].copy()
            if prev_a is not None:
                action_change.append(float(np.linalg.norm(a - prev_a)))
                accel.append(float(np.linalg.norm(v - prev_v)) * C.CTRL_FREQ)
            prev_a, prev_v = a, v
            steps += 1
            done = term or trunc
        guided.append(guided_steps / steps)
        if info["collision"]:
            crashes += 1
            kinds[info["collision_kind"]] += 1
        elif info["success"]:
            successes += 1
            # success fires at the goal radius, so that is the straight-line reference
            ref_m = info["leg_m"] - C.MISSION_GOAL_RADIUS
            extra.append(info["path_m"] / ref_m - 1.0)
            time_ratio.append(info["time_s"] / (ref_m / C.MISSION_CRUISE))
        else:
            failures[info["failure"]] += 1
    env.close()                          # one world per call, closed here: nothing accumulates across calls
    return {"level": level, "legs": legs,
            "crash_rate": crashes / legs, "success_rate": successes / legs,
            "failure_rate": (legs - crashes - successes) / legs,
            "extra_distance": float(np.mean(extra)) if extra else None,
            "time_ratio": float(np.mean(time_ratio)) if time_ratio else None,
            "guided_fraction": float(np.mean(guided)),
            "mean_action_change": float(np.mean(action_change)) if action_change else 0.0,
            "p95_accel_ms2": float(np.percentile(accel, 95)) if accel else 0.0,
            "crash_kinds": dict(kinds), "failures": dict(failures)}


def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def markdown_table(results: list) -> str:
    rows = sorted(results, key=lambda r: (r["level"], r["crash_rate"], -r["success_rate"]))
    out = ["| policy | level | crashes | success | extra distance | time vs straight | guided | crashes by cause |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = "-" if r["time_ratio"] is None else f"{r['time_ratio']:.2f}x"
        out.append(f"| {r['policy']} | {r['level']} | {_pct(r['crash_rate'])} | {_pct(r['success_rate'])} | "
                   f"{_pct(r['extra_distance'])} | {t} | {_pct(r['guided_fraction'])} | {r['crash_kinds'] or '-'} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=POLICIES)
    ap.add_argument("--model", default=None, help="PPO .zip for rl / rl_alone")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--legs", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--out", default=None, help="write the result to this JSON file")
    ap.add_argument("--table", nargs="*", default=None, help="print a markdown table of these result files")
    args = ap.parse_args()
    if args.table is not None:                   # expand patterns here: PowerShell passes "results/mission/*.json" as is
        paths = sorted(f for p in args.table for f in (glob.glob(p) or [p]) if "DONE_" not in f)
        print(markdown_table([json.load(open(p)) for p in paths]))
        return
    r = run_legs(make_policy(args.policy, args.model), args.level, args.legs, args.seed0)
    r["policy"] = args.policy if not args.model else f"{args.policy}:{os.path.basename(os.path.dirname(args.model))}"
    print(json.dumps(r), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
