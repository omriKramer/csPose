import fastai.vision as fv
import torch
from torch import nn

import utils
from models import layers
from models.upernet import get_fpn


class Instructor(fv.Callback):

    def __init__(self, object_tree):
        self.tree = object_tree
        self.obj_loss = nn.BCEWithLogitsLoss()
        self.sampler = utils.BalancingSampler(self.tree.n_obj)
        self.inst = None
        self.train = True

    def on_batch_begin(self, train, last_target, **kwargs):
        obj_gt, part_gt = last_target
        if not train:
            inst = torch.arange(1, self.tree.n_obj)
            self.inst = inst.expand(len(obj_gt), len(inst)).T
            self.train = False
            return

        self.train = True
        inst = []
        for obj_gt_i in obj_gt:
            objects = obj_gt_i.unique()
            if objects[0] == 0:
                objects = objects[1:]
            inst.append(self.sampler(objects))
        inst = torch.stack(inst).to(obj_gt.device)
        self.inst = inst[None]

    def loss(self, pred, obj_gt, part_gt):
        if not self.train:
            return 0

        obj_pred = pred
        obj_gt = obj_gt.squeeze(dim=1)
        pred_size = obj_pred.shape[-2:]
        obj_gt = utils.resize(obj_gt, pred_size)

        inst = self.inst[0]
        obj_mask = obj_gt == inst[:, None, None]
        obj_target = obj_mask * 1.
        foreground = obj_gt != 0
        loss = self.obj_loss(pred[foreground], obj_target[foreground])
        return loss


class CSHead(nn.Module):

    def __init__(self, instructor, tree, weights_encoder='', weights_decoder=''):
        super().__init__()
        self.fpn = get_fpn(tree, weights_encoder=weights_encoder, weights_decoder=weights_decoder)
        fpn_dim = 512
        self.td = nn.Sequential(layers.conv_layer(fpn_dim, fpn_dim),
                                layers.conv_layer(fpn_dim, fpn_dim // 2),
                                fv.conv2d(fpn_dim // 2, 1, ks=1, bias=True))
        self.embedding = fv.embedding(tree.n_obj, fpn_dim)
        self.instructor = instructor

    def forward(self, img):
        features = self.fpn(img)
        out = []
        for inst in self.instructor.inst:
            emb_vec = self.embedding(inst)
            embedded_features = features * emb_vec[..., None, None]
            out.append(self.td(embedded_features))

        out = torch.stack(out, dim=1)
        return out
