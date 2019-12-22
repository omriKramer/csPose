import random

import matplotlib.pyplot as plt

import transform
from datasets.kps import CocoSingleKPS


def test_random_horizontal_flip(data_path):
    keypoints = ['left_eye', 'right_eye']

    def get_transform(flip):
        t = [
            transform.ToTensor(),
            transform.ConvertCocoKps(),

        ]
        if flip:
            t.append(transform.RandomHorizontalFlip(1), )
        t.append(transform.ExtractKeypoints(keypoints))
        return transform.Compose(t)

    coco = CocoSingleKPS.from_data_path(data_path, transforms=get_transform(False), keypoints=keypoints)
    t_coco = CocoSingleKPS.from_data_path(data_path, transforms=get_transform(True), keypoints=keypoints)
    amount = 2
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title('original')
    axs[0, 1].set_title('flipped')
    for i, ax_row in zip(random.sample(range(len(coco)), amount), axs):
        ax_row[0].set_ylabel(i)
        image, kps = coco[i]
        t_image, t_kps = t_coco[i]
        for ax, im, p in (ax_row[0], image, kps), (ax_row[1], t_image, t_kps):
            im = im.permute(1, 2, 0).numpy()
            ax.imshow(im, aspect='equal')
            ax.plot(p[:, 0], p[:, 1], 'ro')

    plt.tight_layout()
    plt.show()
