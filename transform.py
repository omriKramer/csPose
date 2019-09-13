import numpy as np
import torch
from torchvision import transforms as T

from coco_utils import decode_keypoints


def resize_keypoints(keypoints, ratios):
    ratio_h, ratio_w = ratios
    new_keypoints = np.array(keypoints, dtype=np.float32)
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

    def __repr__(self):
        return f'{self.__class__.__name__}({self.size})'


class ToTensor:

    def __init__(self, keys=None):
        self.keys = keys
        self.to_tensor = T.ToTensor()

    def __call__(self, img, target):
        img = self.to_tensor(img)
        if self.keys:
            for key in self.keys:
                target[key] = torch.tensor(target[key])

        return img, target

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        return f'{self.__class__.__name__}({self.transforms})'
