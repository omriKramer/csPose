import numpy as np


class CocoEval:

    def __init__(self, sigmas=None):
        if sigmas is None:
            sigmas = np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

        self.sigmas = sigmas

    def compute_oks(self, gt, dt, area):
        var = (self.sigmas * 2) ** 2
        gx = gt[:, 0]
        gy = gt[:, 1]
        v = gt[:, 2]

        dx = dt[:, 0]
        dy = dt[:, 1]

        d = (gx - dx) ** 2 + (gy - dy) ** 2
        e = d / var / (area + np.spacing(1)) / 2
        e = e[v > 0]
        oks = np.sum(np.exp(-e)) / e.shape[0]
        return oks
