import numpy as np


def image_is_dark_or_light(img, dark_thr=50, light_thr=200):
    rgb_mns = img.mean(axis=(0, 1))
    return np.all(rgb_mns < dark_thr) or np.all(rgb_mns > light_thr)

