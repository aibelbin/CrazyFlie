"""Compare finished experiment runs and pick the best setup for the main run.

    python -m drone_follow.pick_winner [--runs-dir runs] [--prefix exp_]

Reads <run>/final_eval_best.json (200 scenarios per level) and <run>/env.txt (the experiment
switches). Score (lower is better) = mean collision rate on the levels >= 2 that were evaluated
                                     + 0.05 * mean following error (m)
                                     + 0.005 * mean p95 acceleration (m/s^2).
Writes <runs-dir>/winner.json. Also prints a table of every run.
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def load(run_dir: str):
    path = os.path.join(run_dir, "final_eval_best.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        res = json.load(f)
    env = {"DF_ACTION_RATE_LIMIT": "0", "DF_R_TILT": "0"}
    try:
        with open(os.path.join(run_dir, "env.txt")) as f:
            env.update(dict(re.findall(r"(\w+)=([\d.eE+-]+)", f.read())))
    except OSError:
        pass
    return res, env


def levels(res: dict) -> list:
    return sorted(int(k.split("level")[1]) for k in res if k.startswith("ppo/level"))


def score(res: dict) -> float:
    lv = [res[f"ppo/level{l}"] for l in levels(res) if l >= 2]
    return float(np.mean([r["collision_rate"] for r in lv])
                 + 0.05 * np.mean([r["mean_follow_error_m"] for r in lv])
                 + 0.005 * np.mean([r["p95_accel_ms2"] for r in lv]))


def table(rows: list) -> str:
    lv = levels(rows[0][1]) if rows else []
    head = "".join(f"{'L' + str(l):>5}" for l in lv)
    out = [f"{'run':<12}{'rate':>6}{'tilt':>6} | {head} collisions | follow err (last 2 levels) | accel p95 | score"]
    for name, res, env, sc in rows:
        c = [res[f"ppo/level{l}"] for l in lv]
        coll = "".join(f"{100 * x['collision_rate']:>4.0f}%" for x in c)
        out.append(f"{name:<12}{env['DF_ACTION_RATE_LIMIT']:>6}{env['DF_R_TILT']:>6} | {coll}            "
                   f"| {c[-2]['mean_follow_error_m']:.2f} / {c[-1]['mean_follow_error_m']:.2f}"
                   f"       | {c[-2]['p95_accel_ms2']:.1f} / {c[-1]['p95_accel_ms2']:.1f}"
                   f"       | {sc:.3f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--prefix", default="exp_")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.runs_dir, args.prefix + "*"))):
        loaded = load(d)
        if loaded:
            res, env = loaded
            rows.append((os.path.basename(d), res, env, score(res)))
    if not rows:
        print("no finished runs found; falling back to the control setup")
        winner = {"name": "fallback", "rate": "0", "tilt": "0", "score": None}
    else:
        rows.sort(key=lambda r: r[3])
        print(table(rows))
        name, _, env, sc = rows[0]
        winner = {"name": name, "rate": env["DF_ACTION_RATE_LIMIT"], "tilt": env["DF_R_TILT"], "score": sc}
        print(f"\nwinner: {name}  (rate limit {winner['rate']}, tilt penalty {winner['tilt']})")
    with open(os.path.join(args.runs_dir, "winner.json"), "w") as f:
        json.dump(winner, f, indent=2)


if __name__ == "__main__":
    main()
