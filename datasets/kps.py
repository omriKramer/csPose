import itertools
import math

import numpy as np
import torchvision

from coco_utils import decode_keypoints, Coco


def fix_kps(kps, frame):
    x, y, v = decode_keypoints(kps)
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

    right = math.ceil(max(bbox[0] + bbox[2], *seg_x, *kps_x))
    lower = math.ceil(max(bbox[1] + bbox[3], *seg_y, *kps_y))

    return left, upper, right, lower


def fix_bbox(bbox, frame):
    new_bbox = bbox.copy()
    new_bbox[0] -= frame[0]
    new_bbox[1] -= frame[1]
    return new_bbox


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
        img = img.crop(frame)

        target = {
            'id': annotation['id'],
            'image_id': annotation['image_id'],
            'category_id': annotation['category_id'],
            'area': annotation['area'],
            'bbox': fix_bbox(annotation['bbox'], frame),
            'keypoints': list(fix_kps(kps, frame)),
            'num_keypoints': annotation['num_keypoints'],
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
