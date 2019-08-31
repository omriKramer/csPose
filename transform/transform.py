import numpy as np

from coco_utils import decode_keypoints


def resize_keypoints(keypoints, ratios):
    ratio_h, ratio_w = ratios
    new_keypoints = np.array(keypoints, dtype=np.float)
    x, y, v = decode_keypoints(new_keypoints)
    x[v > 0] *= ratio_h
    y[v > 0] *= ratio_w
    return new_keypoints


def resize_boxes(box, ratios):
    ratio_h, ratio_w = ratios
    x, y, width, height = box
    x *= ratio_w
    width *= ratio_w
    y *= ratio_h
    height *= ratio_h
    return x, y, width, height


def resize(img, target, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, img.size))
    target['keypoints'] = resize_keypoints(target['keypoints'], ratios)
    target['bbox'] = resize_boxes(target['boxes'], ratios)
    new_img = img.resize(new_size)
    return new_img, target
