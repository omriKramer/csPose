import numpy as np
import torch

from coco_utils import decode_keypoints


class CocoEval:

    def __init__(self, sigmas=None):
        if sigmas is None:
            sigmas = torch.tensor(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

        self.sigmas = sigmas

    def compute_oks(self, gt, dt, area, device=None):
        var = (self.sigmas.to(device) * 2) ** 2
        gx, gy, v = decode_keypoints(gt)
        dx, dy, _ = decode_keypoints(dt)

        d = (gx - dx) ** 2 + (gy - dy) ** 2
        e = d / var / (area + np.spacing(1)) / 2
        e = e[v > 0]
        oks = torch.sum(torch.exp(-e)) / e.shape[0]
        return oks
