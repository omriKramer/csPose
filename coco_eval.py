import numpy as np
import torch

from coco_utils import decode_keypoints


class CocoEval:

    def __init__(self, sigmas=None):
        if sigmas is None:
            sigmas = torch.tensor(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

        self.sigmas = sigmas

    def compute_oks(self, gt, dt, area):
        if dt.nelement() == 0:
            return 0

        var = (self.sigmas * 2) ** 2
        gx, gy, v = decode_keypoints(gt)
        dx, dy, _ = decode_keypoints(dt)

        d = (gx - dx) ** 2 + (gy - dy) ** 2
        e = d / var / (area + np.spacing(1)) / 2
        e = e[v > 0]
        oks = torch.sum(torch.exp(-e)) / e.shape[0]
        return oks


evaluator = CocoEval()


def batch_oks(outputs, targets):
    oks = [evaluator.compute_oks(gt, dt, area) for gt, dt, area in zip(targets['keypoints'], outputs, targets['area'])]
    return torch.tensor(oks, dtype=torch.float).mean()
