import random

import fastai.vision as fv
import torch
from torch import nn
import torch.nn.functional as F

import utils
from models import nnlayers
from models.upernet import get_fpn


class BCEObjectLoss(nn.Module):

    def __init__(self, instructor, pos_weight=None, softmax=False):
        super().__init__()
        self.instructor = instructor
        self.obj_loss = F.binary_cross_entropy if softmax else F.binary_cross_entropy_with_logits
        self.pos_weight = pos_weight
        self.softmax = softmax

    def forward(self, pred, obj_gt, part_gt):
        if not self.instructor.train:
            return torch.zeros(1)

        return self.loss(pred[0], obj_gt, self.instructor.inst[0])

    def loss(self, pred, gt, inst):
        pred = pred.squeeze(dim=1)
        gt = gt.squeeze(dim=1)
        pred_size = pred.shape[-2:]
        gt = utils.resize(gt, pred_size)

        if self.softmax:
            pred = F.softmax(pred, dim=1)

        obj_mask = gt == inst[:, None, None]
        obj_target = obj_mask.float()
        pred = pred[range(len(pred)), inst]
        has_objs = inst != 0
        obj_target = obj_target[has_objs]
        pred = pred[has_objs]

        pred_flat = pred[has_objs].reshape(-1)[None]
        target_flat = obj_target[has_objs].reshape(-1)[None]
        weight = target_flat * self.pos_weight if self.pos_weight else None
        loss = self.obj_loss(pred_flat, target_flat, weight=weight)
        return loss


class Instructor(fv.Callback):

    def __init__(self, tree, obj_classes=None):
        self.tree = tree

        self.sampler = utils.BalancingSampler(self.tree.n_obj)
        self.inst = None
        self.train = True
        self.obj_classes = obj_classes if obj_classes else list(range(1, self.tree.n_obj))

    def sample_train_inst(self, obj_gt):
        inst = []
        for obj_gt_i in obj_gt:
            objects = obj_gt_i.unique().cpu().tolist()
            objects = set(self.obj_classes).intersection(objects)
            if objects:
                c = self.sampler.sample(list(objects))
            else:
                c = 0
            inst.append(c)
        inst = torch.tensor(inst, device=obj_gt.device)
        self.inst = inst[None]

    def on_batch_begin(self, train, last_target, **kwargs):
        obj_gt, part_gt = last_target
        if not train:
            inst = self.create_val_inst(obj_gt)
            self.inst = inst.expand(len(obj_gt), len(inst)).T
            self.train = False
            return

        self.train = True
        self.sample_train_inst(obj_gt)

    def create_val_inst(self, obj_gt):
        inst = torch.tensor(self.obj_classes, device=obj_gt.device)
        return inst


class FullHeadInstructor(Instructor):

    def __init__(self, tree, obj_classes=None, sample_train=True):
        super().__init__(tree, obj_classes)
        self.sample_train = sample_train

    def sample_train_inst(self, obj_gt):
        if not self.sample_train:
            inst = torch.tensor(self.obj_classes, device=obj_gt.device)
            self.inst = inst.expand(len(obj_gt), len(inst)).T
            return

        inst = []
        for obj_gt_i in obj_gt:
            objects = obj_gt_i.unique().cpu().tolist()
            objects = set(self.obj_classes).intersection(objects)
            if objects:
                c = self.sampler.sample(list(objects))
            else:
                c = random.choice(self.obj_classes)
            inst.append(c)
        inst = torch.tensor(inst, device=obj_gt.device)
        self.inst = inst[None]

    def create_val_inst(self, obj_gt):
        obj_classes = [0] + self.obj_classes
        inst = torch.tensor(obj_classes, device=obj_gt.device)
        return inst


class CSHead(nn.Module):

    def __init__(self, instructor, tree, weights_encoder='', weights_decoder='', emb_op=torch.mul):
        super().__init__()
        self.fpn = get_fpn(tree, weights_encoder=weights_encoder, weights_decoder=weights_decoder)
        fpn_dim = 512
        self.td = nn.Sequential(nnlayers.conv_layer(fpn_dim, fpn_dim // 4),
                                nnlayers.conv_layer(fpn_dim // 4, fpn_dim // 8),
                                fv.conv2d(fpn_dim // 8, 1, ks=1, bias=True))
        self.embedding = fv.embedding(tree.n_obj, fpn_dim)
        self.instructor = instructor
        self.emb_op = emb_op

    def forward(self, img):
        features = self.fpn(img)
        out = []
        for inst in self.instructor.inst:
            emb_vec = self.embedding(inst)
            embedded_features = self.emb_op(features, emb_vec[..., None, None])
            out.append(self.td(embedded_features))

        out = torch.cat(out, dim=1)
        return out


class Head2Clbk(fv.Callback):

    def __init__(self, tree):
        self.tree = tree
        self.sampler = utils.BalancingSampler(tree.n_obj)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        obj_gt, part_gt = last_target
        obj_gt = obj_gt.squeeze(dim=1)
        part_gt = part_gt.squeeze(dim=1)

        if train:
            pred_size = obj_gt.shape[1] // 4, part_gt.shape[2] // 4
            obj_gt = utils.resize(obj_gt, pred_size)
            part_gt = utils.resize(part_gt, pred_size)

        part_gt = self.tree.split_parts_gt(obj_gt, part_gt, mark_in_obj=False)
        if train:
            objects = []
            has_parts = torch.any(part_gt.transpose(0, 1).flatten(start_dim=2) > 0, dim=-1)
            obj_with_parts = torch.tensor(list(self.tree.obj_with_parts), dtype=torch.long)
            for img_parts in has_parts:
                present_objects = obj_with_parts[img_parts]
                if len(present_objects) > 0:
                    objects.append(int(self.sampler.sample(present_objects)))
                else:
                    objects.append(None)
            instruction = [self.tree.obj2idx[o] if o else self.tree.n_obj_with_parts for o in objects]
            instruction = torch.tensor(instruction, dtype=torch.long, device=last_input.device)
        else:
            objects = None
            instruction = None

        return {'last_input': (last_input, objects, instruction), 'last_target': (obj_gt, part_gt)}


class CSHead2(nn.Module):

    def __init__(self, tree, weights_encoder='', weights_decoder='', hidden=2):
        super().__init__()
        self.fpn = get_fpn(tree, weights_encoder=weights_encoder, weights_decoder=weights_decoder)
        fpn_dim = 512
        self.embedding = fv.embedding(tree.n_obj_with_parts + 1, fpn_dim)
        self.td = nn.ModuleList([nnlayers.conv_layer(fpn_dim, fpn_dim) for _ in range(hidden)])
        dims = tree.sections + [tree.n_obj]
        self.heads = nn.ModuleList([fv.conv2d(fpn_dim, dim, ks=1, bias=True) for dim in dims])
        self.bu_start = nn.ModuleList([fv.conv2d(dim, fpn_dim // 2) for dim in dims])
        self.bu_lateral = nn.ModuleList([nnlayers.conv_layer(fpn_dim, fpn_dim // 2) for _ in range(hidden)])
        self.bu = nn.ModuleList(
            [nnlayers.conv_layer(fpn_dim, fpn_dim // 2) for _ in range(hidden - 1)]
            + [nnlayers.conv_layer(fpn_dim, fpn_dim)])
        self.obj_inst = tree.n_obj_with_parts
        self.tree = tree

    def forward(self, img, objects, instruction):
        features = self.fpn(img)
        obj_instruction = torch.tensor(self.obj_inst, dtype=torch.long, device=img.device)
        emb_vec = self.embedding(obj_instruction)
        td = []
        x = features * emb_vec[None, :, None, None]
        for m in self.td:
            x = m(x)
            td.append(x)

        obj_pred = self.heads[-1](x)
        x = self.bu_start[-1](obj_pred)
        for lateral, bu_conv, td_out in zip(self.bu_lateral, self.bu, reversed(td)):
            x = torch.cat([x, lateral(td_out)], dim=1)
            x = bu_conv(x)

        if self.training:
            part_pred = self.pred_one_part(features, x, objects, instruction)
        else:
            part_pred = self.pred_all_parts(features, x, obj_pred)
        return obj_pred, part_pred

    def pred_one_part(self, features, bu, objects, instruction):
        emb_vec = self.embedding(instruction)
        x = features * emb_vec[:, :, None, None] + bu
        for m in self.td:
            x = m(x)
        part_pred = []
        for i in range(len(features)):
            o = objects[i]
            if o is not None:
                m_idx = self.tree.obj2idx[o]
                part_pred.append((o, self.heads[m_idx](x[i][None])))
            else:
                part_pred.append(None)
        return part_pred

    def pred_all_parts(self, features, bu, obj_pred):
        objects = obj_pred.argmax(dim=1)
        img_objects = [o.unique().tolist() for o in objects]
        obj_with_parts = set(self.tree.obj_with_parts)
        img_objects_with_parts = [list(obj_with_parts.intersection(o)) for o in img_objects]

        bs, _, h, w = features.shape
        part_pred = torch.zeros((bs, self.tree.n_parts, h, w), device=features.device)
        for i, predicted_objects in enumerate(img_objects_with_parts):
            if len(predicted_objects) == 0:
                continue
            inst_tensor = torch.tensor([self.tree.obj2idx[o] for o in predicted_objects],
                                       dtype=torch.long, device=features.device)
            emb_vec = self.embedding(inst_tensor)
            img_features = features[i].repeat(len(inst_tensor), 1, 1, 1)
            img_bu = bu[i].repeat(len(inst_tensor), 1, 1, 1)
            x = img_features * emb_vec[:, :, None, None] + img_bu
            for m in self.td:
                x = m(x)
            for o, x_o in zip(predicted_objects, x):
                start, end = self.tree.obj2part_idx[o]
                m_idx = self.tree.obj2idx[o]
                part_pred[i, start:end].copy_(self.heads[m_idx](x_o[None])[0])

        return part_pred


class Head2Loss:

    def __init__(self, tree):
        self.tree = tree
        self.obj_ce = nn.CrossEntropyLoss(ignore_index=0)
        self.part_ce = nn.CrossEntropyLoss()

    def __call__(self, pred, obj_gt, part_gt):
        obj_pred, part_pred = pred
        if not isinstance(part_pred, list):
            return torch.zeros(1)

        obj_loss = self.obj_ce(obj_pred, obj_gt)
        part_loss = []
        for i in range(len(part_pred)):
            if part_pred[i] is None:
                continue
            o, img_part_pred = part_pred[i]
            obj_mask = obj_gt[i] == o
            o_part_gt = part_gt[self.tree.obj2idx[o], i]
            o_part_gt = o_part_gt[obj_mask][None]
            img_part_pred = img_part_pred[:, :, obj_mask]
            part_loss.append(self.part_ce(img_part_pred, o_part_gt))

        loss = obj_loss + sum(part_loss)
        return loss
