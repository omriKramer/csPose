from pathlib import Path

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


def show_image_with_kps(img, keypoints, visible=None):
    if isinstance(img, Image.Image):
        img = np.asarray(img)
    elif isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy()

    if not isinstance(keypoints, list):
        keypoints = list(keypoints.reshape((-1,)))

    if visible is not None:
        keypoints[2::3] = visible[2::3]

    plt.imshow(img)
    plt.axis('off')
    _coco_helper.showAnns([{'keypoints': keypoints, 'category_id': 1}])
    plt.show()
