import json
from pathlib import Path


def filter_ds(coco, image_ids):
    new = coco.copy()
    new['images'] = [image for image in coco['images'] if image['id'] in image_ids]
    new['annotations'] = [an for an in coco['annotations'] if an['image_id'] in image_ids]
    return new


def main(train_size=4, val_size=2):
    data_path = Path('~/weizmann/coco').expanduser()
    ann_file = data_path / 'annotations' / 'person_keypoints_val2017.json'
    with ann_file.open() as f:
        coco = json.load(f)

    total_keypoints = 17
    train = set()
    val = set()

    i = 0
    while len(train) < train_size:
        an = coco['annotations'][i]
        if an['num_keypoints'] == total_keypoints:
            train.add(an['image_id'])
        i += 1

    while len(val) < val_size:
        an = coco['annotations'][i]
        if an['num_keypoints'] == total_keypoints and an['image_id'] not in train:
            val.add(an['image_id'])
        i += 1

    print(f'train image ids: {train}')
    print(f'val image ids {val}')

    for mode, ids in (('train', train), ('val', val)):
        output_file = data_path / 'dev' / 'annotations' / f'person_keypoints_{mode}2017.json'
        ds = filter_ds(coco, ids)
        with output_file.open('w') as f:
            json.dump(ds, f)
            print(f'created {output_file}')


if __name__ == '__main__':
    main()
