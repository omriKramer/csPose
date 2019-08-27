import numpy as np

from coco_utils import decode_keypoints


def resize_image_and_keypoints(img, keypoints, new_size):
    new_keypoints = resize_keypoints(keypoints, img.size, new_size)
    new_img = img.resize(new_size)
    return new_img, new_keypoints


def resize_keypoints(keypoints, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_h, ratio_w = ratios
    new_keypoints = np.array(keypoints, dtype=np.float)
    x, y, v = decode_keypoints(new_keypoints)
    x[v > 0] *= ratio_h
    y[v > 0] *= ratio_w
    return new_keypoints
