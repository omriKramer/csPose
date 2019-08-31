from pathlib import Path

coco_dir = Path('~/weizmann/coco').expanduser()
root = coco_dir / 'val2017/'
ann_file = coco_dir / 'annotations/person_keypoints_val2017.json'
