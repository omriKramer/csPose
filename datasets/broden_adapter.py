from abc import ABC

import fastai.vision as fv
import numpy as np


class BrodenAdapter(ABC):

    def __init__(self, obj_mapping, part_mapping, to_tensor=True):
        self.obj_mapping = np.array(obj_mapping)
        self.part_mapping = np.array(part_mapping)
        self.to_tensor = to_tensor

    def get_obj_mask(self, obj_fn):
        raise NotImplemented

    def get_part_mask(self, part_fn):
        raise NotImplemented

    def open(self, obj_fn, part_fn):
        obj = self.get_obj_mask(obj_fn)
        part = self.get_part_mask(part_fn) if part_fn else None
        # self.get_part_mask might return None
        if part is None:
            part = np.zeros_like(obj)

        obj = self.obj_mapping[obj]
        part = self.part_mapping[part]

        if self.to_tensor:
            obj = fv.pil2tensor(obj, np.float32)
            part = fv.pil2tensor(part, np.float32)
        return obj, part
