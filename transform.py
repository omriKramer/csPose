import random

import numpy as np
import torch
from pycocotools import mask as coco_mask
from torchvision import transforms as T

from coco_utils import decode_keypoints


def resize_keypoints(keypoints, ratios):
    ratio_h, ratio_w = ratios
    new_keypoints = np.array(keypoints, dtype=np.float32)
    x, y, v = decode_keypoints(new_keypoints)
    x *= ratio_w
    y *= ratio_h
    if x.max() >= 256 or y.max() >= 256:
        print(x, force=True)
        print(y, force=True)
        print(ratios, force=True)
        print(keypoints, force=True)
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
            target = {key: torch.tensor((target[key])) for key in self.keys}

        return img, target

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            if "boxes" in target:
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

    def __repr__(self):
        return f'{self.__class__.__name__}({self.prob})'


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        return f'{self.__class__.__name__}({self.transforms})'


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target[0]["image_id"]
        image_id = torch.tensor([image_id])

        anno = target
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {"boxes": boxes, "labels": classes, "masks": masks, "image_id": image_id}
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area.to(dtype=torch.float32)
        target["iscrowd"] = iscrowd

        return image, target

    def __repr__(self):
        return f'{self.__class__.__name__}()'
