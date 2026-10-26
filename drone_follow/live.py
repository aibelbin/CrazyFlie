"""Live 3D demo: the drone flies in real time with real quadrotor physics, following a walking
person through a scene with debris, pedestrians and rival drones. Opens a PyBullet window.

    python -m drone_follow.live --model runs/ppo_ft/final.zip

Keys (click the 3D window first):
    o  drop debris between the drone and the person     r  launch a rival drone at the drone
    p  add a walking pedestrian                         n  new episode
    1  trained policy   2  hand-written reactive policy   3  naive follower (watch it crash)
    c  toggle the follow camera (the mouse always works: drag to orbit, scroll to zoom)
`--headless N` runs N control steps with no window (a quick self-check).

Smoothness: the simulation is stepped at its native 240 Hz in real time, and people, debris and
the camera are interpolated between the 30 Hz control steps, so motion is not choppy.
"""
import argparse
import time

import pybullet as p

from drone_follow import config as C
from drone_follow.drone_visual import DroneVisual
from drone_follow.env import FollowAvoidAviary, SingleDroneWrapper
from drone_follow.policies import naive_follow, reactive_follow
from drone_follow.render import status_from
from drone_follow.scenario import Scenario


def key_char(key: int) -> str:
    """The printable character for a PyBullet key code, or '' for special/unknown keys."""
    return chr(key) if 32 <= key < 127 else ""


STATUS_RGB = {"FOLLOW TARGET": [0.1, 0.6, 0.1], "AVOIDING": [0.9, 0.5, 0.0], "COLLISION": [0.9, 0.1, 0.1]}
SUBSTEPS = C.PYB_FREQ // C.CTRL_FREQ


class Pacer:
    """Hooks PyBullet's physics step: after each 240 Hz step it refreshes the visuals
    (interpolated), moves the camera, and sleeps so the simulation runs in real time."""

    def __init__(self, sim, realtime: bool):
        self.sim, self.realtime = sim, realtime
        self.k, self.visual, self.auto_cam = 0, None, True
        self.focus = None
        self.next_t = time.perf_counter()
        self._orig = p.stepSimulation
        p.stepSimulation = self._paced

    def close(self):
        p.stepSimulation = self._orig

    def begin_control_step(self):
        self.k = 0

    def _paced(self, *args, **kwargs):
        out = self._orig(*args, **kwargs)
        self.k += 1
        sim = self.sim
        sim.scenario.sync_interp(self.k / SUBSTEPS, 1.0 / C.PYB_FREQ)
        if self.visual is not None:
            self.visual.update()
        if self.realtime and self.auto_cam:
            self._follow_camera()
        if self.realtime:
            self.next_t += 1.0 / C.PYB_FREQ
            lag = time.perf_counter() - self.next_t
            if lag < 0:
                time.sleep(-lag)
            elif lag > 0.1:                 # fell far behind (e.g. window dragged): resync
                self.next_t = time.perf_counter()
        return out

    def _follow_camera(self):
        """Track the drone/person midpoint but keep the user's own yaw, pitch and zoom."""
        sim = self.sim
        want = 0.5 * (sim.pos[0] + sim.scenario.target_point)
        self.focus = want if self.focus is None else self.focus + 0.04 * (want - self.focus)
        cam = p.getDebugVisualizerCamera(physicsClientId=sim.CLIENT)
        yaw, pitch, dist = cam[8], cam[9], cam[10]
        p.resetDebugVisualizerCamera(dist, yaw, pitch, self.focus.tolist(), physicsClientId=sim.CLIENT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="trained PPO .zip (needed for policy 1)")
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", type=int, default=0)
    args = ap.parse_args()

    policies = {"reactive": reactive_follow, "naive": naive_follow}
    if args.model:
        from drone_follow.evaluate import load_ppo_policy
        policies["trained"] = load_ppo_policy(args.model)
    current = "trained" if "trained" in policies else "reactive"

    gui = args.headless == 0
    Scenario.DEFER_SYNC = True
    env = SingleDroneWrapper(FollowAvoidAviary(
        gui=gui, difficulty=args.difficulty, seed=args.seed, episode_sec=1e9, realistic=True))
    sim = env.unwrapped
    pacer = Pacer(sim, realtime=gui)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=sim.CLIENT)
        p.resetDebugVisualizerCamera(6.5, 45, -28, [0, 0, 1], physicsClientId=sim.CLIENT)

    label_id, last_pos = None, None

    def new_episode(**kw):
        nonlocal label_id, last_pos
        obs, _ = env.reset(**kw)
        pacer.visual = DroneVisual(sim.CLIENT, int(sim.DRONE_IDS[0]))
        pacer.focus = None
        sim.scenario.sync_interp(1.0, 1.0 / C.PYB_FREQ)
        pacer.visual.update()
        label_id, last_pos = None, sim.pos[0].copy()
        return obs

    obs = new_episode(seed=args.seed)
    hold_until, step, crashes, episodes = 0.0, 0, 0, 0
    try:
        while True:
            if gui and not p.isConnected(physicsClientId=sim.CLIENT):
                print("window closed, exiting")
                break
            if gui:
                for key, state in p.getKeyboardEvents(physicsClientId=sim.CLIENT).items():
                    if not state & p.KEY_WAS_TRIGGERED:
                        continue
                    ch = key_char(key)
                    if ch == "o":
                        sim.scenario.spawn_debris_ahead(sim.pos[0])
                    elif ch == "r":
                        sim.scenario.spawn_rival(sim.pos[0])
                    elif ch == "p":
                        sim.scenario.add_pedestrian()
                    elif ch == "n":
                        obs = new_episode()
                    elif ch == "c":
                        pacer.auto_cam = not pacer.auto_cam
                    elif ch and ch in "123":
                        pick = {"1": "trained", "2": "reactive", "3": "naive"}[ch]
                        if pick in policies:
                            current = pick

            if time.time() < hold_until:                    # pause on a collision
                time.sleep(0.02)
                continue
            if hold_until:
                hold_until = 0.0
                obs = new_episode()

            pacer.begin_control_step()
            obs, _, term, trunc, info = env.step(policies[current](obs))   # paced inside
            status = status_from(info, term)
            step += 1
            if term or trunc:
                episodes += 1
                crashes += int(info["collision"])
                if args.headless:
                    obs = new_episode()
                else:
                    hold_until = time.time() + 1.5
            pos = sim.pos[0]
            if gui:
                p.addUserDebugLine(last_pos, pos, [0.2, 0.4, 1.0], 2.0, lifeTime=4.0,
                                   physicsClientId=sim.CLIENT)
                label_id = p.addUserDebugText(
                    f"{status}  [{current}]", pos + [0, 0, 0.55], STATUS_RGB.get(status, [0, 0, 0]),
                    textSize=1.6, replaceItemUniqueId=label_id if label_id is not None else -1,
                    physicsClientId=sim.CLIENT)
            last_pos = pos.copy()
            if args.headless and step >= args.headless:
                break
    finally:
        pacer.close()
    if args.headless:
        print(f"headless ok: {step} steps, {episodes} episodes, {crashes} collisions ({current})")


if __name__ == "__main__":
    try:
        main()
    except p.error as e:                       # the PyBullet window was closed mid-frame
        if "Not connected" not in str(e):
            raise
        print("window closed, exiting")
