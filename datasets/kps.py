import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


def fix_kps(kps, bbox):
    x, y, v = decode_kps(kps)
    x[v > 0] -= bbox[0]
    y[v > 0] -= bbox[1]
    return kps


def decode_kps(kps):
    x = kps[::3]
    y = kps[1::3]
    v = kps[2::3]
    return x, y, v


def make_bbox(bbox, segmentation, kps):
    segmentation = list(itertools.chain.from_iterable(segmentation))
    seg_x = segmentation[::2]
    seg_y = segmentation[1::2]

    kps_x, kps_y, v = decode_kps(kps)
    kps_x = kps_x[v > 0]
    kps_y = kps_y[v > 0]

    left = math.floor(min(bbox[0], *seg_x, *kps_x))
    upper = math.floor(min(bbox[1], *seg_y, *kps_y))

    right = math.ceil(max(bbox[0] + bbox[2], *seg_x, *kps_x))
    lower = math.ceil(max(bbox[1] + bbox[3], *seg_y, *kps_y))

    return left, upper, right, lower


class CocoSingleKPS(Dataset):

    def __init__(self, root, ann_file):
        coco = COCO(ann_file)
        cat_ids = coco.getCatIds()
        img_ids = coco.getImgIds(catIds=cat_ids)
        ann_ids = coco.getAnnIds(imgIds=img_ids)
        annotations = coco.loadAnns(ann_ids)
        annotations = [an for an in annotations if not an['iscrowd'] and an['num_keypoints'] >= 10]
        self.annotations = annotations
        self.coco = coco
        self.root = Path(root)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        kps = np.array(annotation['keypoints'])
        bbox = make_bbox(annotation['bbox'], annotation['segmentation'], kps)

        imag_details = self.coco.loadImgs(annotation['image_id'])[0]
        path = self.root / imag_details['file_name']
        img = Image.open(path).convert('RGB')

        img = img.crop(bbox)
        kps = fix_kps(kps, bbox)
        return img, kps

    def show_item(self, index):
        img, kps = self[index]
        plt.imshow(np.asarray(img))
        plt.axis('off')
        self.coco.showAnns([{'keypoints': list(kps), 'category_id': 1}])
