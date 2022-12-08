import os
import time

import cv2
import numpy as np


def parse_result(label, count, ratio):
    count_d = dict(zip(label, count))
    ramp_count = count_d.get(21, 0)
    if ramp_count / np.sum(count) > ratio:
        return True
    else:
        return False


def get_ramp(results, img_rgb):
    results = np.array(results[0])
    input_height, input_width = results.shape

    # big trapezoid
    big_closest = np.array([
        [0, int(input_height)],
        [int(input_width),
         int(input_height)],
        [int(0.882 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(0.118 * input_width + .5),
         int(.8 * input_height + .5)]
    ])

    big_future = np.array([
        [int(0.118 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(0.882 * input_width + .5),
         int(.8 * input_height + .5)],
        [int(.765 * input_width + .5),
         int(.66 * input_height + .5)],
        [int(.235 * input_width + .5),
         int(.66 * input_height + .5)]
    ])

    # small trapezoid
    small_closest = np.array([
        [488, int(input_height)],
        [1560, int(input_height)],
        [1391, int(.8 * input_height + .5)],
        [621, int(.8 * input_height + .5)]
    ])

    small_future = np.array([
        [741, int(.66 * input_height + .5)],
        [1275, int(.66 * input_height + .5)],
        [1391, int(.8 * input_height + .5)],
        [621, int(.8 * input_height + .5)]
    ])

    upper_left = np.array([
        [1567, 676],
        [1275, 676],
        [1391, 819],
        [1806, 819]
    ])

    bottom_left = np.array([
        [1806, 819],
        [1391, 819],
        [1560, 1024],
        [2048, 1024]
    ])

    upper_right = np.array([
        [741, 676],
        [481, 676],
        [242, 819],
        [621, 819]
    ])

    bottom_right = np.array([
        [621, 819],
        [242, 819],
        [0, 1024],
        [488, 1024]
    ])

    # _draw_closest_and_future((big_closest, big_future), (small_closest, small_future), img_rgb)

    ramp_location = locate_ramp(small_closest, small_future,
                                upper_left, bottom_left,
                                upper_right, bottom_right,
                                results)

    if not ramp_location:
        ramp_location = "no_ramp"

    return ramp_location


def locate_ramp(small_closest, small_future,
                upper_left, bottom_left,
                upper_right, bottom_right,
                results):

    if has_ramp(results, (small_closest, small_future), 0.9, 0.7):
        return "small_trapezoid"

    right_location = has_ramp(results, (bottom_right, upper_right), 0.4, 0.2)
    if right_location:
        return f"{right_location}_left"

    left_location = has_ramp(results, (bottom_left, upper_left), 0.4, 0.2)
    if left_location:
        return f"{left_location}_right"

    return False


def has_ramp(results, areas, partial_ratio, all_ratio):
    bottom, upper = areas
    input_height, input_width = results.shape

    mask = np.zeros((input_height, input_width), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [bottom], 1)
    label, count = np.unique(results[mask == 1], return_counts=True)
    has_ramp_bottom = parse_result(label, count, partial_ratio)

    mask = np.zeros((input_height, input_width), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [upper], 1)
    label, count = np.unique(results[mask == 1], return_counts=True)
    has_ramp_upper = parse_result(label, count, partial_ratio)

    if has_ramp_bottom:
        return "bottom"
    if has_ramp_upper:
        return "upper"

    mask = np.zeros((input_height, input_width), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [bottom], 1)
    mask = cv2.fillPoly(mask, [upper], 1)
    label, count = np.unique(results[mask == 1], return_counts=True)
    has_ramp = parse_result(label, count, all_ratio)
    if has_ramp:
        return "center"
    else:
        return False


def _draw_closest_and_future(big, small, img_rgb):
    big_closest, big_future = big
    small_closest, small_future = small

    img_array = np.array(img_rgb)
    big_closest_color = [0, 50, 50]
    big_future_color = [0, 69, 0]

    small_closest_color = [0, 100, 100]
    small_future_color = [69, 69, 69]

    height, weight, channel = img_array.shape
    img = np.zeros((height, weight, channel), dtype=np.uint8)
    img = cv2.fillPoly(img, [big_closest], big_closest_color)
    img = cv2.fillPoly(img, [big_future], big_future_color)
    img = cv2.fillPoly(img, [small_closest], small_closest_color)
    img = cv2.fillPoly(img, [small_future], small_future_color)

    img_array = 0.3 * img + img_array

    cv2.imwrite("test.png", img_array)
