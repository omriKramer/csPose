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
            inst = torch.arange(1, self.tree.n_obj, device=obj_gt.device)
            self.inst = inst.expand(len(obj_gt), len(inst)).T
            self.train = False
            return

        self.train = True
        inst = []
        for obj_gt_i in obj_gt:
            objects = obj_gt_i.unique()
            if objects[0] == 0 and len(objects) > 1:
                objects = objects[1:]
            inst.append(self.sampler.sample(objects))
        inst = torch.cat(inst).to(obj_gt.device)
        self.inst = inst[None]

    def loss(self, pred, obj_gt, part_gt):
        if not self.train:
            return torch.zeros(1)

        obj_pred = pred.squeeze(dim=1)
        obj_gt = obj_gt.squeeze(dim=1)
        pred_size = obj_pred.shape[-2:]
        obj_gt = utils.resize(obj_gt, pred_size)

        inst = self.inst[0]
        obj_mask = obj_gt == inst[:, None, None]
        obj_target = obj_mask * 1.
        foreground = obj_gt != 0
        loss = self.obj_loss(obj_pred[foreground], obj_target[foreground])
        return loss


class CSHead(nn.Module):

    def __init__(self, instructor, tree, weights_encoder='', weights_decoder=''):
        super().__init__()
        self.fpn = get_fpn(tree, weights_encoder=weights_encoder, weights_decoder=weights_decoder)
        fpn_dim = 512
        self.td = nn.Sequential(layers.conv_layer(fpn_dim, fpn_dim // 4),
                                layers.conv_layer(fpn_dim // 4, fpn_dim // 8),
                                fv.conv2d(fpn_dim // 8, 1, ks=1, bias=True))
        self.embedding = fv.embedding(tree.n_obj, fpn_dim)
        self.instructor = instructor

    def forward(self, img):
        features = self.fpn(img)
        out = []
        for inst in self.instructor.inst:
            emb_vec = self.embedding(inst)
            embedded_features = features * emb_vec[..., None, None]
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
            instruction = []
            has_parts = torch.any(part_gt.transpose(0, 1).flatten(start_dim=2) > 0)
            obj_with_parts = torch.tensor(list(self.tree.obj_with_parts), dtype=torch.long)
            for img_parts in has_parts:
                present_objects = obj_with_parts[img_parts].tolist()
                if not present_objects:
                    instruction.append(self.tree.n_obj)
                else:
                    instruction.append(self.sampler.sample(present_objects))
        else:
            instruction = None
        return {'last_input': (last_input, instruction), 'last_target': (obj_gt, part_gt)}


class CSHead2(nn.Module):

    def __init__(self, tree, weights_encoder='', weights_decoder=''):
        super().__init__()
        self.fpn = get_fpn(tree, weights_encoder=weights_encoder, weights_decoder=weights_decoder)
        fpn_dim = 512
        self.embedding = fv.embedding(tree.n_obj_with_parts + 1, fpn_dim)
        self.td = nn.ModuleList([layers.conv_layer(fpn_dim, fpn_dim), layers.conv_layer(fpn_dim, fpn_dim)])
        dims = tree.sections + [tree.n_obj]
        self.heads = nn.ModuleList([fv.conv2d(fpn_dim, dim, ks=1, bias=True) for dim in dims])
        self.bu_start = nn.ModuleList([fv.conv2d(dim, fpn_dim // 2) for dim in dims])
        self.bu_lateral = nn.ModuleList([layers.conv_layer(fpn_dim, fpn_dim // 2) for _ in range(2)])
        self.bu = nn.ModuleList([
            layers.conv_layer(fpn_dim, fpn_dim // 2),
            layers.conv_layer(fpn_dim, fpn_dim)])
        self.obj_inst = tree.n_obj

    def forward(self, img, instruction):
        features = self.fpn(img)
        obj_instruction = torch.tensor(self.obj_inst, dtype=torch.long, device=img.device)
        emb_vec = self.embedding(obj_instruction)
        td = []
        x = features * emb_vec[None]
        for m in self.td:
            x = m(x)
            td.append(x)

        x = self.heads[-1](x)
        obj_pred = x
        x = self.bu_start[-1](x)
        for lateral, bu_conv, td_out in zip(self.bu_lateral, self.bu, td):
            x = torch.cat([x, lateral(td_out)], dim=1)
            x = bu_conv(x)

        if self.training:
            part_pred = self.pred_one_part(features, x, instruction)
        else:
            part_pred = self.pred_all_parts(features, x, obj_pred)
        return obj_pred, part_pred

    def pred_one_part(self, features, bu, instruction):
        inst_tensor = torch.tensor(instruction, dtype=torch.long, device=features.device)
        emb_vec = self.embedding(inst_tensor)
        x = features * emb_vec + bu
        for m in self.td:
            x = m(x)
        part_pred = []
        for i in range(len(features)):
            part_inst = instruction[i]
            if part_inst != self.obj_inst:
                part_pred.append((part_inst, self.heads[part_inst](x[i][None])))
            else:
                part_pred.append(None)
        return part_pred

    def pred_all_parts(self, features, bu, obj_pred):
        objects = obj_pred.argmax(dim=1)
        img_objects = [o.unique().tolist() for o in objects]
        obj_wit_parts = set(self.tree.obj_with_parts)
        img_objects_with_parts = [list(obj_wit_parts.intersection(o)) for o in img_objects]

        bs, _, h, w = features.shape
        part_pred = torch.zeros((bs, self.tree.n_parts, h, w), device=features.device)
        for i, predicted_objects in enumerate(img_objects_with_parts):
            if len(predicted_objects) == 0:
                continue
            inst_tensor = torch.tensor(predicted_objects, dtype=torch.long, device=features.device)
            emb_vec = self.embedding(inst_tensor)
            img_features = features[i].repeat(len(inst_tensor), 1, 1, 1)
            img_bu = bu[i].repeat(len(inst_tensor), 1, 1, 1)
            x = img_features * emb_vec + img_bu
            for m in self.td:
                x = m(x)
            for o, x_o in zip(predicted_objects, x):
                start, end = self.tree.obj2part_idx[o]
                part_pred[i, start:end].copy_(self.heads[o](x_o[None]))

        return part_pred


class Head2Loss:

    def __init__(self, tree):
        self.tree = tree
        self.obj_ce = nn.CrossEntropyLoss(ignore_index=0)
        self.part_ce = nn.CrossEntropyLoss()

    def __call__(self, pred, obj_gt, part_gt):
        obj_pred, part_pred = pred

        obj_loss = self.obj_ce(obj_pred, obj_gt)
        part_loss = []
        for i in range(len(part_pred)):
            if part_pred[i] is None:
                continue
            o, img_part_pred = part_pred[i]
            obj_mask = obj_gt[i] == o
            o_part_gt = part_gt[self.tree.obj2idx[o], i]
            o_part_gt = o_part_gt[obj_mask][None]
            img_part_pred = img_part_pred[:, : obj_mask]
            part_loss.append(self.part_ce(img_part_pred, o_part_gt))

        loss = obj_loss + sum(part_loss)
        return loss
