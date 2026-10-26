"""Table of every finished run (experiments and main runs) with crash causes.

    python -m drone_follow.summarize [--runs-dir runs]
"""
import argparse
import glob
import os

from drone_follow.pick_winner import load, score, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(args.runs_dir, "exp_*")) + glob.glob(os.path.join(args.runs_dir, "main_*"))):
        loaded = load(d)
        if loaded:
            rows.append((os.path.basename(d), loaded[0], loaded[1], score(loaded[0])))
    print(table(rows))
    print("\ncrash causes (200 scenarios per level):")
    for name, res, _, _ in rows:
        for l in (1, 2, 3):
            print(f"  {name:<12} L{l}: {res[f'ppo/level{l}'].get('collision_kinds', {})}")
    print("\nreference (200 scenarios, same seeds): ppo_ft 15% / 12% / 22%, rule-based 0% / 2% / 12%")


if __name__ == "__main__":
    main()
