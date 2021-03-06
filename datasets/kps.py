import itertools
import math
from pathlib import Path

import numpy as np
import torchvision

import coco_utils


def fix_kps(kps, frame):
    kps = kps.copy()
    x, y, v = coco_utils.decode_keypoints(kps)
    x[v > 0] -= frame[0]
    y[v > 0] -= frame[1]
    return kps


def make_frame(bbox, segmentation, kps):
    segmentation = list(itertools.chain.from_iterable(segmentation))
    seg_x = segmentation[::2]
    seg_y = segmentation[1::2]

    kps_x, kps_y, v = coco_utils.decode_keypoints(kps)
    kps_x = kps_x[v > 0]
    kps_y = kps_y[v > 0]

    left = math.floor(min(bbox[0], *seg_x, *kps_x))
    upper = math.floor(min(bbox[1], *seg_y, *kps_y))

    right = math.ceil(max(bbox[0] + bbox[2] - 1, *seg_x, *kps_x))
    lower = math.ceil(max(bbox[1] + bbox[3] - 1, *seg_y, *kps_y))

    return left, upper, right + 1, lower + 1


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
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, root, ann_file, transform=None, target_transform=None, transforms=None, keypoints=None):
        super().__init__(root, transforms, transform, target_transform)

        if isinstance(keypoints, str):
            keypoints = [keypoints]
        self.keypoints = keypoints

        self.coco = coco_utils.Coco(root, ann_file)
        annotations = self.coco.get_annotations()
        self.annotations = self._filter_annotations(annotations)

    def _filter_annotations(self, annotations):
        if self.keypoints is None:
            annotations = [an for an in annotations if not an['iscrowd'] and an['num_keypoints'] >= 10]
            return annotations

        indices = [coco_utils.KEYPOINTS.index(keypoint_name) for keypoint_name in self.keypoints]
        filtered = []
        for an in annotations:
            kps = np.array(an['keypoints'])
            x, y, v = coco_utils.decode_keypoints(kps)
            if np.all(v[indices]):
                filtered.append(an)

        return filtered

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
            'iscrowd': annotation['iscrowd'],
            'segmentation': fix_segmentation(annotation['segmentation'], frame)
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @classmethod
    def from_data_path(cls, data_path, train=True, **kwargs):
        data_path = Path(data_path).expanduser()
        image_set = 'train' if train else 'val'
        root = data_path / '{}2017'.format(image_set)
        ann_file = data_path / 'annotations/person_keypoints_{}2017.json'.format(image_set)
        return cls(root, ann_file, **kwargs)


class SingleKeypoints(CocoSingleKPS):
    def __init__(self, root, ann_file, transform=None, target_transform=None, transforms=None, keypoints=None):
        super().__init__(root, ann_file, transform, target_transform, transforms, keypoints)
        single_keypoint_anns = []
        for ann in self.annotations:
            kps = np.array(ann['keypoints'])
            visible = kps[2:3]
            visible_idx = visible.nonzero()[0]
            all_keypoints = [(ann, idx) for idx in visible_idx]
            single_keypoint_anns.extend(all_keypoints)

        self.annotations = single_keypoint_anns

    def __getitem__(self, index):
        annotation, keypoints_idx = self.annotations[index]
        kps = np.array(annotation['keypoints'])
        frame = make_frame(annotation['bbox'], annotation['segmentation'], kps)

        img = self.coco.get_img(annotation['image_id'])
        img = img.crop(frame)

        kps = list(fix_kps(kps, frame))
        start = keypoints_idx * 3
        target = kps[start:start + 3]

        inp = img, keypoints_idx
        if self.transforms is not None:
            inp, target = self.transforms(inp, target)

        return inp, kps
