import itertools
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class LipPose(torch.utils.data.Dataset):
    c = ['R_Ankle',
         'R_Knee',
         'R_Hip',
         'L_Hip',
         'L_Knee',
         'L_Ankle',
         'B_Pelvis',
         'B_Spine',
         'B_Neck',
         'B_Head',
         'R_Wrist',
         'R_Elbow',
         'R_Shoulder',
         'L_Shoulder',
         'L_Elbow',
         'L_Wrist']

    def __init__(self, image_folder, ann_file, transform=None):

        self.images = list(image_folder.iterdir())
        self.anns = self._read_anns(ann_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image_file = self.images[i]
        img = Image.open(image_file)
        keypoints = self.anns.loc[image_file.name].values
        keypoints = torch.tensor(keypoints).reshape(-1, 3)

        if self.transform:
            img, keypoints = self.transform(img, keypoints)

        return img, keypoints

    @classmethod
    def train_val(cls, root, train_transform=None, val_transform=None):
        root = Path(root)
        datasets = []
        for phase, transform in ('train', train_transform), ('val', val_transform):
            image_folder = root / f'{phase}_images'
            an_file = root / 'pose_annotations' / f'lip_{phase}_set.csv'
            datasets.append(cls(image_folder, an_file, transform))

        return datasets

    def _read_anns(self, ann_file):
        names = [(f'x_{p}', f'y_{p}', f'v_{p}') for p in self.c]
        names = list(itertools.chain.from_iterable(names))
        anns = pd.read_csv(ann_file, header=None, names=names, index_col=0)
        v_columns = [col for col in names if col.startswith('v')]
        anns[v_columns] += 1
        anns = anns.fillna(0)
        return anns
