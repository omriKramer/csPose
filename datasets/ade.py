import json
import fastai.vision as fv
import numpy as np
from imageio import imread

from datasets.broden_adapter import BrodenAdapter


def ade_decode(im):
    """Decodes pixel-level object/part class and instance data from
    the given image, previously encoded into RGB channels."""
    # Classes are a combination of RG channels (dividing R by 10)
    return (im[:, :, 0] // 10) * 256 + im[:, :, 1]


def ade_load_seg(fn):
    im = imread(fn)
    seg = ade_decode(im)
    return seg


class AdeAdapter(BrodenAdapter):
    def __init__(self, root):
        self.root = root
        with (self.root / 'ade_index_mapping.json').open() as f:
            to_broden = json.load(f)

        super().__init__(to_broden['object'], to_broden['part'])

    def get_obj_mask(self, obj_fn):
        obj = ade_load_seg(obj_fn)
        return obj

    def get_part_mask(self, part_fn):
        part = ade_load_seg(part_fn)
        return part

