import json

import numpy as np
from pycocotools.coco import COCO

import coco_utils
from datasets import kps
from transform import resize_keypoints

with open('../coco/keypoints_rcnn_val.json') as f:
    results = json.load(f)

results = {int(k): {'keypoints': np.array(v['keypoints']), 'scores': v['scores']} for k, v in results.items()}
coco = COCO('/Volumes/waic/shared/coco/annotations/person_keypoints_val2017.json')

sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
IMAGE_SIZE = 128, 128


def compute_oks(dt, ann):
    area = ann['area']
    gt = ann['keypoints']

    var = (sigmas * 2) ** 2
    v = gt[:, 2]
    dt = dt[:, :2]
    gt = gt[:, :2]

    d = np.linalg.norm(dt - gt, axis=1)
    e = d / var / (area + np.spacing(1)) / 2
    e = e[v > 0]
    oks = np.sum(np.exp(-e)) / e.shape[0]
    return oks


def match_detections(detections, gt):
    assert all(detections['scores'][i] >= detections['scores'][i + 1] for i in range(len(detections['scores']) - 1))
    matches = []
    unmatched = {anno['id']: anno for anno in gt}
    for dt in detections['keypoints']:
        if not unmatched:
            break
        best_match = max(unmatched, key=lambda ann_id: compute_oks(dt, unmatched[ann_id]))
        matches.append((dt, unmatched[best_match]))
        del unmatched[best_match]

    return matches


def distance(dt, ann, original_size):
    frame = kps.make_frame(ann['bbox'], ann['segmentation'], ann['keypoints'].reshape(-1))
    ratios = [original_size[i] / IMAGE_SIZE[i] for i in range(2)]

    dt = kps.fix_kps(dt.reshape(-1), frame)
    dt = resize_keypoints(dt, ratios, IMAGE_SIZE).reshape(-1, 3)

    gt = kps.fix_kps(ann['keypoints'].reshape(-1), frame)
    gt = resize_keypoints(gt, ratios, IMAGE_SIZE).reshape(-1, 3)

    v = gt[:, 2]
    dt = dt[:, :2]
    gt = gt[:, :2]
    res = np.linalg.norm((dt - gt), axis=1)
    assert len(res) == 17
    res[v == 0] = None
    res = np.append(res, np.nanmean(res))
    return res


def main():
    matches = []
    for image_id in results:
        detections = results[image_id]
        ann_ids = coco.getAnnIds([image_id])
        gt = coco.loadAnns(ann_ids)
        for ann in gt:
            ann['keypoints'] = np.array(ann['keypoints']).reshape(-1, 3)

        gt = [ann for ann in gt if not np.all(ann['keypoints'] == 0)]

        image_matches = match_detections(detections, gt)
        matches.extend(image_matches)

    distances = []
    for detection, ann in matches:
        image = coco.loadImgs([ann['image_id']])[0]
        original_size = image['height'], image['width']
        distances.append(distance(detection, ann, original_size))

    distances = np.array(distances)
    distances = np.nanmean(distances, axis=0)
    cat = coco_utils.KEYPOINTS.copy()
    cat.append('mean_distance')
    return dict(zip(cat, distances))


print(main())

# {'nose': 14.78428, 'left_eye': 15.316996, 'right_eye': 15.375622, 'left_ear': 14.923258, 'right_ear': 16.169493,
#  'left_shoulder': 17.448637, 'right_shoulder': 19.353815, 'left_elbow': 13.72023, 'right_elbow': 17.824326,
#  'left_wrist': 15.288893, 'right_wrist': 17.951794, 'left_hip': 15.128732, 'right_hip': 16.5737,
#  'left_knee': 14.553558, 'right_knee': 16.662119, 'left_ankle': 15.416061, 'right_ankle': 16.400297,
#  'mean_distance': 20.220213}
