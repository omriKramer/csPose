import itertools
import math
from pathlib import Path

import numpy as np
import torchvision

from coco_utils import decode_keypoints, Coco


def fix_kps(kps, frame):
    x, y, v = decode_keypoints(kps.copy())
    x[v > 0] -= frame[0]
    y[v > 0] -= frame[1]
    return kps


def make_frame(bbox, segmentation, kps):
    segmentation = list(itertools.chain.from_iterable(segmentation))
    seg_x = segmentation[::2]
    seg_y = segmentation[1::2]

    kps_x, kps_y, v = decode_keypoints(kps)
    kps_x = kps_x[v > 0]
    kps_y = kps_y[v > 0]

    left = math.floor(min(bbox[0], *seg_x, *kps_x))
    upper = math.floor(min(bbox[1], *seg_y, *kps_y))

    right = math.ceil(max(bbox[0] + bbox[2] - 1, *seg_x, *kps_x))
    lower = math.ceil(max(bbox[1] + bbox[3] - 1, *seg_y, *kps_y))

    return left, upper, right, lower


def fix_bbox(bbox, frame):
    new_bbox = bbox.copy()
    new_bbox[0] -= frame[0]
    new_bbox[1] -= frame[1]
    return new_bbox


def fix_segmentation(segmentation, frame):
    fixed_segmentation = []
    for seg in segmentation:
        seg = np.array(seg, dtype=np.float32)
        seg[::2] -= frame[0]
        seg[1::2] -= frame[1]
        fixed_segmentation.append(seg.tolist())
    return fixed_segmentation


class CocoSingleKPS(torchvision.datasets.VisionDataset):

    def __init__(self, root, ann_file, transform=None, target_transform=None, transforms=None):
        super().__init__(root, transforms, transform, target_transform)

        coco = Coco(root, ann_file)
        cat_ids = coco.getCatIds()
        img_ids = coco.getImgIds(catIds=cat_ids)
        ann_ids = coco.getAnnIds(imgIds=img_ids)
        annotations = coco.loadAnns(ann_ids)
        annotations = [an for an in annotations if not an['iscrowd'] and an['num_keypoints'] >= 10]
        self.annotations = annotations
        self.coco = coco

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        kps = np.array(annotation['keypoints'])
        frame = make_frame(annotation['bbox'], annotation['segmentation'], kps)

        img = self.coco.get_img(annotation['image_id'])
        original_size = img.size
        img = img.crop(frame)

        target = {
            'id': annotation['id'],
            'image_id': annotation['image_id'],
            'category_id': annotation['category_id'],
            'area': annotation['area'],
            'bbox': fix_bbox(annotation['bbox'], frame),
            'keypoints': list(fix_kps(kps, frame)),
            'num_keypoints': annotation['num_keypoints'],
            'iscrowd': annotation['iscrowd'],
            'segmentation': fix_segmentation(annotation['segmentation'], frame)
        }
        x, y, v = decode_keypoints(target['keypoints'])
        w, h = img.size
        if x.max() >= w:
            print(annotation['keypoints'])
            print(x)
            print(frame)
            print(original_size)
            assert False
        if y.max() >= h:
            print(annotation['keypoints'])
            print(y)
            print(frame)
            print(original_size)
            assert False

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @classmethod
    def from_data_path(cls, data_path, train=True, transform=None, target_transform=None, transforms=None):
        data_path = Path(data_path).expanduser()
        image_set = 'train' if train else 'val'
        root = data_path / '{}2017'.format(image_set)
        ann_file = data_path / 'annotations/person_keypoints_{}2017.json'.format(image_set)
        return cls(root, ann_file, transform=transform, target_transform=target_transform, transforms=transforms)
