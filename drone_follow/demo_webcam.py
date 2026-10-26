"""Live demo: webcam -> YOLO person detection -> simulated drone follows and avoids hazards.

    python -m drone_follow.demo_webcam --model runs/ppo/final.zip

Keys:  o = drop debris between the drone and you     r = launch a rival drone at it
       n = new episode                               q = quit
`--source` also accepts a video/image path, and `--save out.mp4 --no-window --max-frames N`
runs headless (useful without a camera).
"""
import argparse
import time

import cv2
import numpy as np

from drone_follow import config as C
from drone_follow.env import FollowAvoidAviary, SingleDroneWrapper
from drone_follow.evaluate import load_ppo_policy
from drone_follow.person_tracker import PersonTracker, box_to_world
from drone_follow.policies import reactive_follow
from drone_follow.render import STATUS_COLORS, render_frame, status_from


def open_source(src: str):
    if src.isdigit():
        return cv2.VideoCapture(int(src)), False
    img = cv2.imread(src)
    if img is not None:                      # a still image: reuse it every frame
        return img, True
    return cv2.VideoCapture(src), False


def read(cap, still):
    if still:
        return True, cap.copy()
    return cap.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="trained PPO .zip (default: reactive baseline)")
    ap.add_argument("--source", default="0", help="camera index, video path, or image path")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--person-height", type=float, default=0.7)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    policy = load_ppo_policy(args.model) if args.model else reactive_follow
    tracker = PersonTracker(args.weights)
    env = SingleDroneWrapper(FollowAvoidAviary(difficulty=0, seed=0, episode_sec=1e9))
    sim = env.unwrapped
    obs, _ = env.reset(seed=0)

    cap, still = open_source(args.source)
    writer, frame_idx = None, 0
    status = "NO TARGET"
    while True:
        t0 = time.time()
        ok, frame = read(cap, still)
        if not ok:
            break
        box = tracker.detect(frame)
        if box is not None:
            h, w = frame.shape[:2]
            sim.scenario.set_person(
                box_to_world(box, w, h, args.hfov, args.person_height), 1.0 / C.CTRL_FREQ)
            cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (60, 200, 60), 2)
        # The sim only advances while a person is locked; otherwise the drone holds position.
        if box is not None:
            obs, _, term, trunc, info = env.step(policy(obs))
            status = status_from(info, term)
            if term or trunc:
                obs, _ = env.reset()
        else:
            status = "NO TARGET"

        view = render_frame(sim, 640, 480, status=status, title="Drone view (simulated)")
        cam = cv2.resize(frame, (640, 480))
        cv2.putText(cam, "FOLLOW TARGET" if box is not None else "SEARCHING...", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    STATUS_COLORS["FOLLOW TARGET" if box is not None else "NO TARGET"], 3, cv2.LINE_AA)
        canvas = np.hstack([cam, view])
        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         C.CTRL_FREQ, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
        if not args.no_window:
            cv2.imshow("Detection & Avoidance", canvas)
            key = cv2.waitKey(max(1, int(1000 / C.CTRL_FREQ - (time.time() - t0) * 1000))) & 0xFF
            if key == ord("q"):
                break
            if key == ord("o"):
                sim.scenario.spawn_debris_ahead(sim.pos[0])
            if key == ord("r"):
                sim.scenario.spawn_rival(sim.pos[0])
            if key == ord("n"):
                obs, _ = env.reset()
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break
    if writer:
        writer.release()
    if not still:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
