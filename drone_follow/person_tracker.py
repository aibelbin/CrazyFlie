"""Pretrained YOLO person detection, and the mapping from a camera box to a world position.

The webcam stands in for the drone's camera. A detected person is turned into a position
in the simulated world, which the trained policy then follows.
"""
from dataclasses import dataclass

import numpy as np

PERSON_CLASS = 0     # COCO class id


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class PersonTracker:
    """Wraps a pretrained Ultralytics YOLO model; returns the largest person in frame."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.4):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame_bgr: np.ndarray) -> Box | None:
        result = self.model.predict(frame_bgr, classes=[PERSON_CLASS], conf=self.conf,
                                    verbose=False)[0]
        boxes = [Box(*b.xyxy[0].tolist(), float(b.conf[0])) for b in result.boxes]
        return max(boxes, key=lambda b: b.height, default=None)


def box_to_world(box: Box, frame_w: int, frame_h: int, hfov_deg: float = 60.0,
                 person_height_m: float = 0.7, origin_depth_m: float = 2.0,
                 limit: float = 3.5) -> np.ndarray:
    """World (x, y) of the person: x is away from the camera, y is to the camera's left.

    Depth comes from the box height (pinhole model). `person_height_m` is the visible
    height of the person: ~0.7 m for head-and-torso at a desk, ~1.7 m for a full body.
    The result is re-centred so a person at `origin_depth_m` maps to x = 0.
    """
    focal_px = (frame_w / 2) / np.tan(np.radians(hfov_deg) / 2)
    depth = np.clip(person_height_m * focal_px / max(box.height, 1.0), 0.5, 8.0)
    lateral = (box.cx - frame_w / 2) / focal_px * depth        # + = right of the image
    xy = np.array([depth - origin_depth_m, -lateral])
    return np.clip(xy, -limit, limit)
