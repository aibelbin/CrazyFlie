"""Pictures of one mission leg for a presentation: top and side view, every policy's path on the same leg.

    python -m drone_follow.mission_viz --level 2 --pick --out videos/mission_L2.png --gif videos/mission_L2.gif
    python -m drone_follow.mission_viz --level 3 --seed 30005 --policies straight bendy rl --model runs/mission_s1/best.zip

Every policy flies the same seeded leg (as in mission_evaluate). Thick line = the avoider in control (GUIDED),
thin line = AUTO, cross = crash. Light work: a second or two per policy, fine on a laptop.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")                         # files only, no window: works the same on Windows, macOS and Linux
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from drone_follow import config as C
from drone_follow.mission_env import make_mission_env
from drone_follow.mission_evaluate import EVAL_SEED0, make_policy

COLORS = {"straight": "#7f7f7f", "potential": "#e69f00", "bendy": "#0072b2", "rl": "#009e73", "rl_alone": "#cc79a7"}
LABELS = {"straight": "straight line (no avoidance)", "potential": "potential field", "bendy": "BendyRuler-style",
          "rl": "RL (PPO) + supervisor", "rl_alone": "RL (PPO) alone"}
KIND_COLORS = {"tree": "#2e7d32", "pole": "#333333", "rock": "#8d6e63", "building": "#90a4ae", "wall": "#6d4c41"}
NEAR_PATH = 8.0                               # m, the side view shows obstacles this close to the straight line


def fly(policy, level: int, seed: int, length: float | None = None):
    """Fly one seeded leg. Returns (leg, positions (T, 3), guided flags (T,), final info)."""
    env = make_mission_env(level=level, seed=seed, length=length)()
    sim = env.unwrapped
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    pos, guided, done = [sim.pos[0].copy()], [False], False
    while not done:
        obs, _, term, trunc, info = env.step(np.clip(np.asarray(policy(obs), float).reshape(3), -1.0, 1.0))
        pos.append(sim.pos[0].copy())
        guided.append(bool(getattr(policy, "guided", False)))
        done = term or trunc
    leg = sim.leg
    env.close()
    return leg, np.array(pos), np.array(guided), info


def outcome(info: dict) -> str:
    if info["collision"]:
        return f"crashed ({info['collision_kind']})"
    if info["success"]:
        return f"reached in {info['time_s']:.0f} s"
    return info["failure"] or "stopped"


def pick_seed(level: int, seed0: int = EVAL_SEED0, tries: int = 30, length: float | None = None) -> int:
    """First seeded leg where the straight line crashes and BendyRuler gets through: a leg worth showing."""
    for seed in range(seed0, seed0 + tries):
        if fly(make_policy("straight"), level, seed, length)[3]["collision"] \
                and fly(make_policy("bendy"), level, seed, length)[3]["success"]:
            return seed
    return seed0


def _draw_world(top, side, leg):
    for (x, y, z, r), kind in zip(leg.spheres, leg.sphere_kind):          # canopies and rocks
        top.add_patch(Circle((x, y), r, color=KIND_COLORS[kind], alpha=0.35, lw=0))
        if abs(y) < NEAR_PATH:
            side.add_patch(Circle((x, z), r, color=KIND_COLORS[kind], alpha=0.35, lw=0))
    for (x, y, z0, z1, r), kind in zip(leg.cylinders, leg.cylinder_kind):  # trunks and poles
        top.add_patch(Circle((x, y), max(r, 0.25), color=KIND_COLORS[kind], lw=0))
        if abs(y) < NEAR_PATH:
            side.add_patch(Rectangle((x - r, z0), 2 * r, z1 - z0, color=KIND_COLORS[kind], lw=0))
    for (x0, y0, z0, x1, y1, z1), kind in zip(leg.boxes, leg.box_kind):  # buildings and walls
        top.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=KIND_COLORS[kind], alpha=0.8, lw=0))
        if y0 < NEAR_PATH and y1 > -NEAR_PATH:
            side.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, color=KIND_COLORS[kind], alpha=0.8, lw=0))
    xs = np.linspace(-10, leg.length + 10, 300)
    ground = leg.terrain.height(xs, np.zeros_like(xs))
    side.fill_between(xs, ground.min() - 1.0, ground, color="#c5b358", alpha=0.5, lw=0)
    for ax, (sx, sy), (gx, gy) in ((top, leg.start[:2], leg.goal[:2]), (side, leg.start[[0, 2]], leg.goal[[0, 2]])):
        ax.plot(sx, sy, "ko", ms=6)
        ax.plot(gx, gy, "*", color="#d62728", ms=16, zorder=5)
        ax.add_patch(Circle((gx, gy), C.MISSION_GOAL_RADIUS, fill=False, ls="--", color="#d62728"))


def _setup_axes(leg, flights):
    fig, (top, side) = plt.subplots(2, 1, figsize=(14, 7.5), height_ratios=[1.6, 1])
    paths = np.concatenate([pos for pos, _, _ in flights.values()])
    half = max(12.0, float(np.abs(paths[:, 1]).max()) + 4.0)
    zs = paths[:, 2]
    for ax in (top, side):
        ax.set_xlim(-8, leg.length + 8)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    top.set_ylim(-half, half)
    side.set_ylim(float(leg.terrain.height(0.0, 0.0)) - 1.0, max(float(zs.max()), float(leg.goal[2])) + 6.0)
    top.set_ylabel("y (m)")
    side.set_ylabel("height (m)")
    side.set_xlabel("distance along the leg (m)")
    _draw_world(top, side, leg)
    return fig, top, side


def plot(leg, flights: dict, out: str, title: str):
    """flights: policy name -> (positions, guided, info). Writes a PNG."""
    fig, top, side = _setup_axes(leg, flights)
    for name, (pos, guided, info) in flights.items():
        c = COLORS.get(name, "k")
        for ax, k in ((top, 1), (side, 2)):
            ax.plot(pos[:, 0], pos[:, k], color=c, lw=1.2, label=f"{LABELS.get(name, name)}: {outcome(info)}" if k == 1 else None)
            g = np.where(guided, pos[:, k], np.nan)
            ax.plot(pos[:, 0], g, color=c, lw=4, alpha=0.8, solid_capstyle="round")
            if info["collision"]:
                ax.plot(pos[-1, 0], pos[-1, k], "X", color=c, ms=13, mec="k", zorder=6)
    top.legend(loc="upper left", fontsize=9, framealpha=0.9)
    top.set_title(title)
    side.set_title(f"side view (obstacles within {NEAR_PATH:.0f} m of the straight line); thick = avoider in control",
                   fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def animate(leg, flights: dict, out: str, title: str, every: int = 6, fps: int = 20):
    """Top and side view with each drone moving along its path. Writes a GIF (every `every` control steps a frame)."""
    fig, top, side = _setup_axes(leg, flights)
    top.set_title(title)
    artists = []
    for name, (pos, guided, info) in flights.items():
        c = COLORS.get(name, "k")
        trail_t, = top.plot([], [], color=c, lw=1.5, label=LABELS.get(name, name))
        dot_t, = top.plot([], [], "o", color=c, ms=8, mec="k")
        trail_s, = side.plot([], [], color=c, lw=1.5)
        dot_s, = side.plot([], [], "o", color=c, ms=8, mec="k")
        artists.append((pos, guided, info, trail_t, dot_t, trail_s, dot_s))
    top.legend(loc="upper left", fontsize=9, framealpha=0.9)
    clock = top.text(0.99, 0.95, "", transform=top.transAxes, ha="right", va="top", fontsize=10)
    n_frames = max(len(a[0]) for a in artists) // every + 2
    fig.tight_layout()

    def frame(i):
        t = i * every
        for pos, guided, info, trail_t, dot_t, trail_s, dot_s in artists:
            j = min(t, len(pos) - 1)
            trail_t.set_data(pos[: j + 1, 0], pos[: j + 1, 1])
            trail_s.set_data(pos[: j + 1, 0], pos[: j + 1, 2])
            crashed = info["collision"] and j == len(pos) - 1
            for dot, k in ((dot_t, 1), (dot_s, 2)):
                dot.set_data([pos[j, 0]], [pos[j, k]])
                dot.set_marker("X" if crashed else "o")
                dot.set_markersize(13 if crashed or guided[j] else 8)
        clock.set_text(f"t = {t / C.CTRL_FREQ:4.1f} s")
        return []

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    FuncAnimation(fig, frame, frames=n_frames).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2, help="0 empty, 1 sparse trees, 2 dense trees + poles, 3 + buildings and a wall")
    ap.add_argument("--seed", type=int, default=None, help="leg seed (default: the first evaluation leg)")
    ap.add_argument("--pick", action="store_true", help="search for a leg where the straight line crashes and BendyRuler gets through")
    ap.add_argument("--policies", nargs="+", default=["straight", "potential", "bendy"])
    ap.add_argument("--model", default=None, help="PPO .zip, needed for the rl / rl_alone policies")
    ap.add_argument("--length", type=float, default=None, help="leg length in m (default: random 80-150)")
    ap.add_argument("--out", default="videos/mission.png")
    ap.add_argument("--gif", default=None, help="also write an animation here")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else (pick_seed(args.level, length=args.length) if args.pick else EVAL_SEED0)
    flights, leg = {}, None
    for name in args.policies:
        leg, pos, guided, info = fly(make_policy(name, args.model), args.level, seed, args.length)
        flights[name] = (pos, guided, info)
        print(f"{name:10s} {outcome(info)}", flush=True)
    title = f"Mission leg, level {args.level} (seed {seed}): {leg.length:.0f} m at {C.MISSION_CRUISE:.0f} m/s"
    plot(leg, flights, args.out, title)
    print(f"wrote {args.out}")
    if args.gif:
        animate(leg, flights, args.gif, title)
        print(f"wrote {args.gif}")


if __name__ == "__main__":
    main()
