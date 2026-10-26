import numpy as np

from drone_follow.person_tracker import Box, box_to_world

W, H = 640, 480


def test_centred_person_maps_to_zero_lateral():
    xy = box_to_world(Box(270, 100, 370, 400, 0.9), W, H)
    assert abs(xy[1]) < 1e-6


def test_person_on_image_right_is_negative_y():
    xy = box_to_world(Box(500, 100, 600, 400, 0.9), W, H)
    assert xy[1] < 0


def test_bigger_box_means_closer():
    near = box_to_world(Box(270, 50, 370, 450, 0.9), W, H)
    far = box_to_world(Box(270, 200, 370, 300, 0.9), W, H)
    assert near[0] < far[0]


def test_result_is_clipped_to_arena():
    xy = box_to_world(Box(0, 0, 10, 5, 0.9), W, H)       # tiny box -> very far
    assert np.all(np.abs(xy) <= 3.5)
