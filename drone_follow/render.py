"""Offscreen 3D rendering of the simulation (works without a GUI window)."""
import cv2
import numpy as np
import pybullet as p

from drone_follow import config as C

STATUS_COLORS = {          # BGR
    "FOLLOW TARGET": (60, 200, 60),
    "AVOIDING": (0, 165, 255),
    "COLLISION": (40, 40, 230),
    "NO TARGET": (160, 160, 160),
}


def render_frame(env, width: int = 640, height: int = 480, status: str = "", title: str = "") -> np.ndarray:
    """BGR image from a camera hovering behind-and-above the drone/person pair."""
    c = env.CLIENT
    drone = env.pos[0]
    person = env.scenario.target_point
    focus = 0.5 * (drone + person)
    eye = focus + np.array([-4.5, -4.5, 3.2])
    view = p.computeViewMatrix(eye.tolist(), focus.tolist(), [0, 0, 1], physicsClientId=c)
    proj = p.computeProjectionMatrixFOV(60.0, width / height, 0.1, 60.0, physicsClientId=c)
    _, _, rgba, _, _ = p.getCameraImage(width, height, view, proj,
                                        renderer=p.ER_TINY_RENDERER, physicsClientId=c)
    img = cv2.cvtColor(np.reshape(rgba, (height, width, 4)).astype(np.uint8), cv2.COLOR_RGBA2BGR)
    if title:
        cv2.rectangle(img, (0, 0), (width, 42), (40, 40, 40), -1)
        cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if status:
        color = STATUS_COLORS.get(status, (255, 255, 255))
        cv2.putText(img, status, (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return img


def status_from(info: dict, terminated: bool) -> str:
    if terminated and info.get("collision"):
        return "COLLISION"
    return "AVOIDING" if info.get("min_range", C.MAX_RANGE) < 1.3 else "FOLLOW TARGET"
