import json

import numpy as np
import torch
from imageio import imread


def ade_decode(im):
    """Decodes pixel-level object/part class and instance data from
    the given image, previously encoded into RGB channels."""
    # Classes are a combination of RG channels (dividing R by 10)
    return (im[:, :, 0] // 10) * 256 + im[:, :, 1]


def ade_load_seg(fn):
    im = imread(fn)
    seg = ade_decode(im)
    return seg


class AdeAdapter:
    def __init__(self, root):
        self.root = root
        with (self.root / 'ade_index_mapping.json').open() as f:
            self.ade2broden = json.load(f)
            self.ade2broden = {k: np.array(v) for k, v in self.ade2broden.items()}

    def open_ade(self, object_fn, parts_fn):
        obj_seg = ade_load_seg(object_fn)
        if parts_fn:
            parts_seg = ade_load_seg(parts_fn)
        else:
            parts_seg = np.zeros_like(obj_seg)

        obj_seg = self.ade2broden['object'][obj_seg]
        parts_seg = self.ade2broden['part'][parts_seg]
        return torch.from_numpy(obj_seg)[None], torch.from_numpy(parts_seg)[None]
