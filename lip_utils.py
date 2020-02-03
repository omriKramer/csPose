import numpy as np

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
colors = np.array([(255, 0, 0), (255, 0, 0),
                   (0, 255, 0), (0, 255, 0),
                   (0, 0, 255), (0, 0, 255),
                   (0, 0, 255), (128, 128, 0),
                   (128, 128, 0), (128, 0, 128),
                   (128, 0, 128)]) / 255


def plot_joint(ax, joints, visible, annotate=False):
    xs = [j[0].item() for j, v in zip(joints, visible) if v]
    ys = [j[1].item() for j, v in zip(joints, visible) if v]
    params = {'s': 20, 'marker': '.', 'c': 'r'}
    ax.scatter(xs, ys, **params)
    if annotate:
        ans = [name for name, v in zip(c, visible) if v]
        for x, y, name in zip(xs, ys, ans):
            ax.annotate(name, (x, y))

    for segment_idx, color, in zip(bombs, colors):
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
