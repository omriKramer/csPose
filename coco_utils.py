from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
             'left_shoulder', 'right_shoulder', 'left_elbow',
             'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
             'right_hip', 'left_knee', 'right_knee', 'left_ankle',
             'right_ankle']

_coco_helper = COCO()
_coco_helper.cats = {1: {'id': 1,
                         'keypoints': KEYPOINTS,
                         'name': 'person',
                         'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                                      [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                                      [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],
                         'supercategory': 'person'}}


class Coco(COCO):

    def __init__(self, img_dir, ann_file):
        self.image_dir = Path(img_dir)
        super().__init__(ann_file)

    def show_by_id(self, img_id):
        img = self.get_img(img_id)
        annotations = self.get_annotations(img_id)

        plt.imshow(np.asarray(img))
        plt.axis('off')
        self.showAnns(annotations)
        plt.show()

    def get_img(self, img_id):
        img_id = [img_id]
        img_details = self.loadImgs(img_id)[0]
        img = Image.open(self.image_dir / img_details['file_name']).convert('RGB')
        return img

    def get_annotations(self, img_id):
        ann_ids = self.getAnnIds(imgIds=[img_id])
        annotations = self.loadAnns(ann_ids)
        return annotations


def decode_keypoints(keypoints):
    x = keypoints[::3]
    y = keypoints[1::3]
    v = keypoints[2::3]
    return x, y, v


def collate_fn(batch):
    return tuple(zip(*batch))


def plot_image_with_kps(img, keypoints, visible=None, ax: Optional[plt.Axes] = None):
    if ax is None:
        ax = plt.gca()

    ax.tick_params(axis=u'both', which=u'both', length=0)
    if isinstance(img, Image.Image):
        img = np.asarray(img)
    elif isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()

    if torch.is_tensor(keypoints):
        keypoints = keypoints.reshape((-1,)).numpy()
    elif isinstance(keypoints, list):
        keypoints = np.array(keypoints)
    if visible is not None:
        keypoints[2::3] = visible[2::3]

    ax.imshow(img)
    if keypoints.size == 0:
        return

    x, y, v = decode_keypoints(keypoints)
    c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
    sks = np.array(_coco_helper.cats[1]['skeleton']) - 1
    for sk in sks:
        if np.all(v[sk] > 0):
            ax.plot(x[sk], y[sk], linewidth=3, color=c)
    ax.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k', markeredgewidth=2)
    ax.plot(x[v > 1], y[v > 1], 'o', markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
