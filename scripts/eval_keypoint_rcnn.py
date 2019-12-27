import json

import numpy as np
from pycocotools.coco import COCO

import coco_utils

with open('../coco/keypoints_rcnn_val.json') as f:
    results = json.load(f)

results = {int(k): {'keypoints': np.array(v['keypoints']), 'scores': v['scores']} for k, v in results.items()}
coco = COCO('/Volumes/waic/shared/coco/annotations/person_keypoints_val2017.json')

sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


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


def distance(dt, ann):
    gt = ann['keypoints']
    v = gt[:, 2]
    dt = dt[:, :2]
    gt = gt[:, :2]
    res = np.linalg.norm((dt - gt), axis=1)
    assert len(res) == 17
    res[v == 0] = None
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

    distances = [distance(detection, ann) for detection, ann in matches]
    distances = np.array(distances)
    distances = np.nanmean(distances, axis=0)
    return dict(zip(coco_utils.KEYPOINTS, distances))


print(main())
#
# {'nose': 25.430961373399665, 'left_eye': 25.356712569779198, 'right_eye': 26.229808541150103,
#  'left_ear': 24.023034601320667, 'right_ear': 27.828080306797183, 'left_shoulder': 33.266949965986704,
#  'right_shoulder': 33.62888398110362, 'left_elbow': 30.065678281119123, 'right_elbow': 31.892165830442195,
#  'left_wrist': 35.77970678647127, 'right_wrist': 35.61240224901763, 'left_hip': 36.73840208764587,
#  'right_hip': 36.698881362383524, 'left_knee': 33.27601767317778, 'right_knee': 34.53142868747248,
#  'left_ankle': 35.78155198948934, 'right_ankle': 35.86249350592035}
