from enum import Enum

import numpy as np


class KeypointGroup:
    def __init__(self, s):
        self.keypoints = s

    @classmethod
    def create(cls, *args):
        return cls(set(args))

    def join(self, other):
        return self.__class__(self.keypoints | other.keypoints)

    @property
    def indices(self):
        ind = [c.index(k) for k in self.keypoints]
        ind.sort()
        return ind

    def __repr__(self):
        return f'{self.__class__}(keypoints={self.keypoints})'


class BasicGroups(Enum):
    HEAD = KeypointGroup.create('B_Neck', 'B_Head')
    SHOULDERS = KeypointGroup.create('R_Shoulder', 'L_Shoulder')
    KNEES = KeypointGroup.create('R_Knee', 'L_Knee')
    ANKLES = KeypointGroup.create('R_Ankle', 'L_Ankle')
    WRISTS = KeypointGroup.create('R_Wrist', 'L_Wrist')
    ELBOWS = KeypointGroup.create('R_Elbow', 'L_Elbow')
    SPINE = KeypointGroup.create('B_Spine', 'B_Pelvis')


stats = [0.2682, 0.2387, 0.2219], [0.3046, 0.2805, 0.2727]
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

bombs = [[0, 1], [1, 2],
         [3, 4], [4, 5],
         [6, 7], [7, 8],
         [8, 9], [10, 11],
         [11, 12], [13, 14],
         [14, 15]]
line_colors = np.array([(255, 0, 0), (255, 0, 0),
                        (0, 255, 0), (0, 255, 0),
                        (0, 0, 255), (0, 0, 255),
                        (0, 0, 255), (128, 128, 0),
                        (128, 128, 0), (128, 0, 128),
                        (128, 0, 128)]) / 255


def plot_joint(ax, joints, visible, annotate=False, plot_lines=True, colors='r'):
    xs = [j[0].item() for j, v in zip(joints, visible) if v]
    ys = [j[1].item() for j, v in zip(joints, visible) if v]

    params = {'s': 20, 'marker': '.', 'c': colors}
    ax.scatter(xs, ys, **params)

    if annotate:
        ans = [name for name, v in zip(c, visible) if v]
        for x, y, name in zip(xs, ys, ans):
            ax.annotate(name, (x, y))

    if plot_lines:
        for segment_idx, color, in zip(bombs, line_colors):
            first = segment_idx[0]
            x1 = joints[first, 0].item()
            y1 = joints[first, 1].item()
            v1 = visible[first]

            second = segment_idx[1]
            x2 = joints[second, 0].item()
            y2 = joints[second, 1].item()
            v2 = visible[second]

            if v1 and v2:
                ax.plot([x1, x2], [y1, y2], c=color, linewidth=4)
