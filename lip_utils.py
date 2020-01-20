import numpy as np

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


def plot_joint(ax, joints, visible):
    for segment_idx, color, in zip(bombs, colors):
        first = segment_idx[0]
        x1 = joints[first, 0].item()
        y1 = joints[first, 1].item()
        v1 = visible[first]

        second = segment_idx[1]
        x2 = joints[second, 0].item()
        y2 = joints[second, 1].item()
        v2 = visible[second]

        xs, ys = [], []
        if v1:
            xs.append(x1)
            ys.append(y1)
        if v2:
            xs.append(x2)
            ys.append(y2)

        params = {'s': 20, 'marker': '.', 'c': 'r'}
        ax.scatter(xs, ys, **params)
        if v1 and v2:
            ax.plot(xs, ys, c=color, linewidth=4)
