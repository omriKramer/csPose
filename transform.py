import numpy as np
import torch
from torchvision import transforms as T

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
    target['bbox'] = resize_boxes(target['bbox'], ratios)
    target['area'] *= ratios[0] * ratios[1]
    new_img = img.resize(new_size)
    return new_img, target


class ResizeKPS:

    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class ToTensor:

    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, img, target):
        img = self.to_tensor(img)

        for t in ('keypoints', 'area'):
            if t in target:
                target[t] = torch.tensor(target[t], dtype=torch.float)

        return img, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
