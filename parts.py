import random
from collections import OrderedDict
from pathlib import Path

import fastai.vision as fv
import pandas as pd
import torch
from fastai.vision import imagenet_stats
from torch import nn

import models.cs_v2 as cs
import utils
from models import layers


class ObjectTree:

    def __init__(self, tree, obj_names, part_names):
        self.tree = OrderedDict(sorted(tree.items(), key=lambda item: item[0]))
        self.obj2idx = {o: i for i, o in enumerate(self.tree.keys())}
        self.obj_names = obj_names
        self.part_names = part_names

        self.obj2part_idx = {}
        start = 0
        for n_parts, o in zip(self.sections, self.obj_with_parts):
            end = start + n_parts
            self.obj2part_idx[o] = start, end
            start = end

    @classmethod
    def from_meta_folder(cls, meta):
        meta = Path(meta)
        tree = pd.read_csv(Path(meta) / 'object_part_hierarchy.csv')
        tree = {row.object_label: row.part_labels.split(';') for row in tree.itertuples()}
        tree = {k: [int(o) for o in v] for k, v in tree.items()}

        objects = pd.read_csv(meta / 'object.csv')
        objects = objects['name'].tolist()

        parts = pd.read_csv(meta / 'part.csv')
        parts = parts['name'].tolist()

        return cls(tree, objects, parts)

    def __getitem__(self, item):
        return self.tree[item]

    def obj_and_parts(self, names=False):
        if not names:
            return self.tree.items()

        return [(self.obj_names[o], [self.part_names[p] for p in parts])
                for o, parts in self.tree.items()]

    @property
    def obj_with_parts(self):
        return self.tree.keys()

    @property
    def n_obj(self):
        return len(self.obj_names)

    @property
    def n_obj_with_parts(self):
        return len(self.obj_with_parts)

    @property
    def n_parts(self):
        return sum(len(parts) for parts in self.tree.values())

    @property
    def sections(self):
        sections = [len(parts) for parts in self.tree.values()]
        return sections

    def split_parts_pred(self, t: fv.Tensor):
        """
        Split parts prediction by objects.
        Args:
            t: Tensor of shape: (bs, n_parts, h , w)

        Returns: List of length n_obj_with_parts where each item is tensor of shape (bs, n_parts_i, h, w)

        """
        return t.split(self.sections, dim=1)

    def get_part_pred(self, t):
        if t.ndim == 3:
            t = t[None]
        part_pred_list = self.split_parts_pred(t)
        pred = torch.stack([o.argmax(dim=1) for o in part_pred_list])
        return pred.squeeze()

    def split_parts_gt(self, obj: fv.Tensor, part: fv.Tensor, mark_in_obj=True):
        """
        Splits parts gt by objects.
        Args:
            obj: Tensor of shape: (bs, h , w)
            part: Tensor of shape: (bs, h , w)
            mark_in_obj: bool, non-object pixels will be marked as -1

        Returns: Tensor of shape (n_obj_with_parts, bs, h, w)

        """

        present_obj = obj.unique().cpu().tolist()
        present_obj = [o for o in present_obj if o in self.tree]
        classes = torch.tensor(list(self.obj_with_parts), device=obj.device)
        obj_masks = obj == classes[:, None, None, None]
        parts_inside_obj = part[None] * obj_masks
        gt = torch.full_like(parts_inside_obj, -1 if mark_in_obj else 0)
        for o in present_obj:
            obj_parts = self.tree[o]
            i = self.obj2idx[o]
            inside_obj_i = parts_inside_obj[i]
            for part_idx, part in enumerate(obj_parts[1:], start=1):
                part_mask = inside_obj_i == part
                gt[i][part_mask] = part_idx

        if not mark_in_obj:
            return gt

        # if an object has parts then label background (non-part) pixels inside the object with 0
        is_part = gt > 0
        has_parts = is_part.flatten(start_dim=2).any(dim=2, keepdim=True)[..., None]
        bg_inside_obj = has_parts * obj_masks * (~is_part)
        gt[bg_inside_obj] = 0
        return gt

    def cs_preds_func(self, last_output):
        obj_pred, part_pred_dict = last_output
        obj_pred = obj_pred.argmax(dim=1)

        bs, h, w = obj_pred.shape
        n_obj_with_parts = self.n_obj_with_parts
        part_pred = torch.zeros((n_obj_with_parts, bs, h, w), dtype=torch.long, device=obj_pred.device)
        for o, p_pred in part_pred_dict.items():
            has_obj = torch.flatten(obj_pred == o, start_dim=1).any(dim=1)
            p_pred = p_pred.argmax(dim=1) * has_obj[:, None, None]
            part_pred[self.obj2idx[o]] = p_pred

        return obj_pred, part_pred


class ObjectAndParts(fv.ItemBase):

    def __init__(self, objects: fv.ImageSegment, parts: fv.ImageSegment):
        self.objects = objects
        self.parts = parts

    @property
    def data(self):
        return self.objects.data, self.parts.data

    def apply_tfms(self, tfms, **kwargs):
        objects = self.objects.apply_tfms(tfms, **kwargs)
        parts = self.parts.apply_tfms(tfms, **kwargs)
        return self.__class__(objects, parts)

    def __repr__(self):
        return f'{self.__class__.__name__} {tuple(self.objects.size)}'


def _encode_from_labels(t, labels):
    out = t.clone()
    for i, l in enumerate(labels):
        out[t == l] = i
    return out


def _encode(t1, t2):
    labels = torch.stack((t1, t2)).unique()
    out1 = _encode_from_labels(t1, labels)
    out2 = _encode_from_labels(t2, labels)
    return out1, out2, len(labels) - 1


def encode_colors(o1, o2):
    obj1, part1, = o1.data
    obj2, part2 = o2.data

    obj1, obj2, vmax_obj = _encode(obj1, obj2)
    part1, part2, vmax_part = _encode(part1, part2)

    encoded1 = ObjectAndParts(fv.ImageSegment(obj1), (fv.ImageSegment(part1)))
    encoded2 = ObjectAndParts(fv.ImageSegment(obj2), (fv.ImageSegment(part2)))
    return encoded1, encoded2, (vmax_obj, vmax_part)


class ObjectsPartsLabelList(fv.ItemList):

    def __init__(self, items, tree: ObjectTree = None, **kwargs):
        super().__init__(items, **kwargs)
        self.tree = tree

    def get(self, i):
        object_fn, parts_fn = super().get(i)
        obj = fv.open_mask(object_fn, convert_mode='I')
        if parts_fn:
            parts = fv.open_mask(parts_fn, convert_mode='L')
        else:
            parts = fv.ImageSegment(torch.zeros_like(obj.px))
        return ObjectAndParts(obj, parts)

    def analyze_pred(self, pred, scale=4):
        obj, part = pred[:self.tree.n_obj], pred[self.tree.n_obj:]
        size = [s * scale for s in obj.shape[-2:]]
        obj = utils.resize(obj, size)
        part = utils.resize(part, size)

        obj = obj.argmax(dim=0)
        part = self.tree.get_part_pred(part)
        part_agg = torch.zeros_like(obj)
        for (o, parts), part_pred in zip(self.tree.obj_and_parts(), part):
            parts = torch.tensor(parts)
            part_pred = parts[part_pred]
            part_agg = part_agg.where(obj != o, part_pred)
        return obj, part_agg

    def reconstruct(self, t, x=None):
        obj, parts = t
        if obj.ndim == 2:
            obj = obj[None]
            parts = parts[None]
        obj = fv.ImageSegment(obj)
        parts = fv.ImageSegment(parts)
        return ObjectAndParts(obj, parts)


def restrict_to_labeled(pred: ObjectAndParts, gt: ObjectAndParts, tree: ObjectTree = None):
    obj_pred, part_pred = pred.data
    obj_gt, part_gt = gt.data
    obj_pred[obj_gt == 0] = 0
    if tree:
        classes = torch.tensor(list(tree.obj_with_parts))[:, None, None]
        is_obj = obj_gt == classes
        is_obj_and_part = is_obj * (part_gt > 0)
        obj_has_parts = is_obj_and_part.flatten(start_dim=1).any(dim=1)
        obj_with_parts_mask = is_obj * obj_has_parts[:, None, None]
        part_pred = part_pred * obj_with_parts_mask.sum(dim=0, keepdim=True)

    return ObjectAndParts(fv.ImageSegment(obj_pred), fv.ImageSegment(part_pred))


class ObjectsPartsItemList(fv.ImageList):
    _label_cls = ObjectsPartsLabelList
    _square_show = False
    _square_show_res = False

    def show_xys(self, xs, ys, imgsize=4, figsize=None, overlay=True, **kwargs):
        rows = len(xs)
        if overlay:
            axs = fv.subplots(rows, 2, imgsize=imgsize, figsize=figsize)
            for x, y, ax_row in zip(xs, ys, axs):
                x.show(ax=ax_row[0], y=y.objects, **kwargs)
                x.show(ax=ax_row[1], y=y.parts, **kwargs)
        else:
            axs = fv.subplots(rows, 3, imgsize=imgsize, figsize=figsize)
            for x, y, ax_row in zip(xs, ys, axs):
                x.show(ax=ax_row[0], **kwargs)
                y.objects.show(ax=ax_row[1], alpha=1, **kwargs)
                y.parts.show(ax=ax_row[2], alpha=1, **kwargs)

    def show_xyzs(self, xs, ys, zs, tree=None, imgsize=4, figsize=None, **kwargs):
        rows = len(xs)
        axs = fv.subplots(rows, 4, imgsize=imgsize, figsize=figsize)
        for x, y, z, ax_row in zip(xs, ys, zs, axs):
            z = restrict_to_labeled(z, y, tree=tree)
            y, z, vmax = encode_colors(y, z)

            x.show(ax=ax_row[0], y=y.objects, vmax=vmax[0], **kwargs)
            x.show(ax=ax_row[1], y=z.objects, vmax=vmax[0], **kwargs)

            x.show(ax=ax_row[2], y=y.parts, vmax=vmax[1], **kwargs)
            x.show(ax=ax_row[3], y=z.parts, vmax=vmax[1], **kwargs)

        titles = 'objet-GT', 'object-Pred', 'part-GT', 'part-pred'
        for ax, t in zip(axs[0], titles):
            ax.set_title(t)


def pix_acc(pred, gt):
    non_bg = gt > 0
    correct = (pred == gt) * non_bg
    correct = correct.sum()
    total = non_bg.sum()
    return correct, total


def iou(pred, gt, n_classes, mask):
    # ignore pixels outside mask (gt is already inside mask)
    pred = pred * mask
    classes = torch.arange(1, n_classes, device=pred.device)[:, None, None, None]
    # {pred\gt}_i shape: (n_classes-1, bs, h, w)
    pred_i = pred == classes
    gt_i = gt == classes
    intersection = torch.sum(pred_i * gt_i, dim=(1, 2, 3))
    union = torch.sum(pred_i + gt_i, dim=(1, 2, 3))
    return intersection, union


def to_cpu(x):
    if torch.is_tensor(x):
        return x.cpu()
    return torch.tensor(x, device='cpu')


class Accuracy:
    def __init__(self, n=1):
        self.correct = torch.zeros(n)
        self.total = torch.zeros(n)

    def update(self, correct, total):
        self.correct += to_cpu(correct)
        self.total += to_cpu(total)

    def accuracy(self):
        c = self.correct.float()
        # some parts have 0 pixels in the validation set, add epsilon for now
        t = self.total.float() + 1e-10
        return torch.mean(c / t).item()


class BrodenMetrics:
    def __init__(self, obj_tree: ObjectTree, restrict=True, object_only=False):
        self.obj_tree = obj_tree
        self.restrict = restrict
        self.object_only = object_only

        self.obj_pa = Accuracy()
        self.obj_iou = Accuracy(self.obj_tree.n_obj - 1)
        self.part_pa = Accuracy()
        self.part_iou = [Accuracy(n - 1) for n in self.obj_tree.sections]

    def update(self, obj_gt, part_gt, obj_pred, part_pred):
        obj_gt = obj_gt.squeeze(dim=1)
        part_gt = part_gt.squeeze(dim=1)
        if len(part_gt) != self.obj_tree.n_obj_with_parts:
            part_gt = self.obj_tree.split_parts_gt(obj_gt, part_gt)

        gt_size = obj_gt.shape[-2:]
        obj_pred = utils.resize(obj_pred, gt_size)

        obj_pred = obj_pred.argmax(dim=1)

        self.obj_pa.update(*pix_acc(obj_pred, obj_gt))
        self.obj_iou.update(*iou(obj_pred, obj_gt, self.obj_tree.n_obj, obj_gt > 0))

        if self.object_only:
            return

        part_pred = utils.resize(part_pred, gt_size)
        part_pred = self.obj_tree.split_parts_pred(part_pred)
        # part_pred shape: (n_obj_with_parts, bs, h, w)
        part_pred = torch.stack([obj_parts.argmax(dim=1) for obj_parts in part_pred], dim=0)
        if self.restrict:
            part_pred = self.restrict_part_to_obj(obj_pred, part_pred)

        self.part_pa.update(*pix_acc(part_pred, part_gt))
        for i, (obj, parts) in enumerate(self.obj_tree.obj_and_parts()):
            mask = part_gt[i] > -1
            self.part_iou[i].update(*iou(part_pred[i], part_gt[i], len(parts), mask))

    def restrict_part_to_obj(self, obj_pred, part_pred):
        objects_with_parts = list(self.obj_tree.obj_with_parts)
        objects_with_parts = torch.tensor(objects_with_parts, device=obj_pred.device)[:, None, None, None]
        object_pred_mask = obj_pred == objects_with_parts
        part_pred = part_pred * object_pred_mask
        return part_pred

    def avg(self):
        parts_iou = [c.accuracy() for c in self.part_iou]
        parts_iou = sum(parts_iou) / len(parts_iou)
        results = [
            self.obj_pa.accuracy(),
            self.obj_iou.accuracy(),
            self.part_pa.accuracy(),
            parts_iou
        ]
        return results


class BrodenMetricsClbk(utils.LearnerMetrics):

    def __init__(self, learn, obj_tree: ObjectTree, split_func=None, restrict=True):
        super().__init__(learn, ['object-P.A.', 'object-mIoU', 'part-P.A.', 'part-mIoU(bg)'])
        self.split_func = split_func
        self.obj_tree = obj_tree
        self.restrict = restrict

    def _reset(self):
        self.metrics = BrodenMetrics(self.obj_tree, restrict=self.restrict)

    def on_batch_end(self, last_output, last_target, train, **kwargs):
        if train:
            return

        obj_gt, part_gt = last_target
        if self.split_func:
            obj_pred, part_pred = self.split_func(last_output)
        else:
            obj_pred, part_pred = last_output[:, :self.obj_tree.n_obj], last_output[:, self.obj_tree.n_obj:]

        self.metrics.update(obj_gt, part_gt, obj_pred, part_pred)

    def on_epoch_end(self, last_metrics, **kwargs):
        results = self.metrics.avg()
        return fv.add_metrics(last_metrics, results)


class Loss:

    def __init__(self, object_tree: ObjectTree, split_func=None):
        self.object_tree = object_tree
        # unlabeled pixels are zeros
        self.obj_ce = nn.CrossEntropyLoss(ignore_index=0)
        # outside objects gt values are expected to be -1, background inside objects is 0
        self.part_ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.split = split_func

    def __call__(self, pred, obj_gt, part_gt):
        if self.split:
            obj_pred, part_pred = self.split(pred)
        else:
            obj_pred, part_pred = pred

        obj_gt = obj_gt.squeeze(dim=1)
        part_gt = part_gt.squeeze(dim=1)

        pred_size = obj_pred.shape[-2:]
        obj_gt = utils.resize(obj_gt, pred_size)
        part_gt = utils.resize(part_gt, pred_size)
        part_gt = self.object_tree.split_parts_gt(obj_gt, part_gt)

        obj_loss = self.obj_ce(obj_pred, obj_gt)
        part_loss = []
        for o, o_part_pred in part_pred.items():
            i = self.object_tree.obj2idx[o]
            o_part_gt = part_gt[i]
            if torch.any(o_part_gt > -1):
                part_loss.append(self.part_ce(o_part_pred, o_part_gt))

        loss = obj_loss + sum(part_loss)
        return loss


def precision_recall(pred, gt):
    pred = pred.flatten(start_dim=1)
    gt = gt.flatten(start_dim=1)
    true = pred == gt
    false = ~true
    negative = ~pred

    tp = (true * pred).sum(dim=1)
    fp = (false * pred).sum(dim=1)
    tn = (true * negative).sum(dim=1)
    fn = (false * negative).sum(dim=1)
    return tp, fp, tn, fn


class BinaryBrodenMetrics(utils.LearnerMetrics):

    def __init__(self, learn, obj_tree: ObjectTree, thresh=0.75):
        names = ['object-P.A.', 'object-mIoU', 'overlap', 'no_class', f'precision@{thresh:.2}', f'recall@{thresh:.2}']
        super().__init__(learn, names)
        self.tree = obj_tree
        self.thresh = thresh

    def _reset(self):
        self.metrics = BrodenMetrics(self.tree, object_only=True)
        self.precision = Accuracy(self.tree.n_obj - 1)
        self.recall = Accuracy(self.tree.n_obj - 1)
        self.overlap = Accuracy()
        self.no_class = Accuracy()

    def on_batch_end(self, last_output, last_target, train, **kwargs):
        if train:
            return

        obj_gt, part_gt = last_target
        obj_gt = obj_gt.squeeze(dim=1)
        if isinstance(last_output, tuple):
            obj_pred, part_pred = last_output
        else:
            obj_pred = last_output
            part_pred = None

        size = obj_gt.shape[-2:]
        obj_pred = utils.resize(obj_pred, size)
        binary_pred = obj_pred.transpose(0, 1).sigmoid() > self.thresh

        objects = torch.arange(1, self.tree.n_obj, device=obj_gt.device)
        binary_gt = obj_gt == objects[:, None, None, None]
        tp, fp, tn, fn = precision_recall(binary_pred, binary_gt)
        self.precision.update(tp, tp + fp)
        self.recall.update(tp, tp + fn)

        num_classes = binary_pred.sum(dim=0)
        overlapping = (num_classes > 1).bool()
        self.overlap.update(overlapping.sum(), overlapping.numel())
        no_class = num_classes < 1
        self.no_class.update(no_class.sum(), no_class.numel())

        obj_combined = torch.cat([torch.zeros((len(obj_pred), 1, *size), device=obj_pred.device), obj_pred], dim=1)
        self.metrics.update(obj_gt, part_gt, obj_combined, part_pred)

    def on_epoch_end(self, last_metrics, **kwargs):
        results = self.metrics.avg()[:2]
        results.extend(x.accuracy() for x in (self.overlap, self.no_class, self.precision, self.recall))
        return fv.add_metrics(last_metrics, results)


class Labeler:

    def __init__(self, ann_folder):
        self.ann_folder = ann_folder

    def __call__(self, item):
        stem = Path(item).stem
        obj_seg = self.ann_folder / f'{stem}_obj.png'
        part_seg = self.ann_folder / f'{stem}_part.png'
        if not part_seg.exists():
            part_seg = None

        return obj_seg, part_seg


def get_data(broden_root, tree=None, size=256, norm_stats=fv.imagenet_stats, padding_mode='zeros',
             do_flip=True, max_rotate=10., max_zoom=1.1, max_lighting=0.2, max_warp=0.2,
             p_affine=0.75, p_lighting=0.75, **databunch_kwargs):
    labeler = Labeler(broden_root / 'reindexed2')
    tfms = fv.get_transforms(do_flip=do_flip, max_rotate=max_rotate, max_zoom=max_zoom, max_lighting=max_lighting,
                             max_warp=max_warp, p_affine=p_affine, p_lighting=p_lighting)

    data = (ObjectsPartsItemList.from_csv(broden_root, 'trainval.csv')
            .split_from_df(col='is_valid')
            .label_from_func(labeler, tree=tree)
            .transform(tfms, tfm_y=True, size=size, resize_method=fv.ResizeMethod.PAD, padding_mode=padding_mode)
            .databunch(**databunch_kwargs))
    if norm_stats:
        data = data.normalize(norm_stats)
    return data


class TDHead(nn.Module):

    def __init__(self, in_channels, n_objects, parts_sections):
        super().__init__()
        self.conv = layers.conv_layer(in_channels, in_channels)
        module_list = [fv.conv2d(in_channels, n_parts, ks=1, bias=True) for n_parts in parts_sections]
        module_list.append(fv.conv2d(in_channels, n_objects, ks=1, bias=True))
        self.classifier = nn.ModuleList(module_list)

    def forward(self, x, o):
        out = self.conv(x)
        out = self.classifier[o](out)
        return out


class CsNet(nn.Module):
    def __init__(self, body, obj_tree: ObjectTree, sample_one=False, emb_op=torch.mul):
        super().__init__()
        self.sample_one = sample_one
        td_head_ni = body[0].out_channels
        td_head = TDHead(td_head_ni, obj_tree.n_obj, obj_tree.sections)
        self.ifn, self.bu, self.td, self.td_head, self.laterals, channels = cs.create_bu_td(body, td_head)
        self.embedding = fv.embedding(obj_tree.n_obj, channels[-1])
        self.obj_tree = obj_tree
        self.emb_op = emb_op

    def forward(self, img, gt=None):
        bs = img.shape[0]
        features = self.ifn(img)

        x = self.bu(features)
        obj_inst = torch.zeros(bs, dtype=torch.long, device=img.device)
        emb = self.embedding(obj_inst)[..., None, None]
        x = self.emb_op(x, emb)
        obj_pred = self.td(x)
        obj_pred = self.td_head(obj_pred, -1)

        if self.training:
            obj_gt = gt[0]
            objects = obj_gt.unique()
        else:
            objects = obj_pred.argmax(dim=1).unique()

        objects_int = objects.tolist()
        objects = [(o, o_int) for o, o_int in zip(objects, objects_int) if o_int in self.obj_tree.obj_with_parts]
        if self.sample_one and self.training:
            objects = random.sample(objects, 1)

        x = self.bu(features)
        part_pred = {}
        for o, o_int in objects:
            emb = self.embedding(o)[..., None, None]
            td_in = self.emb_op(x, emb)
            o_idx = self.obj_tree.obj2idx[o_int]
            out = self.td(td_in)
            out = self.td_head(out, o_idx)
            part_pred[o_int] = out

        self.clear()
        return obj_pred, part_pred

    def clear(self):
        for lateral in self.laterals:
            del lateral.origin_out
            lateral.origin_out = None


def part_learner(data, arch, obj_tree: ObjectTree,
                 pretrained=False, sample_one=False, emb_op=torch.mul,
                 **learn_kwargs):
    body = fv.create_body(arch, pretrained)
    model = CsNet(body, obj_tree, sample_one=sample_one, emb_op=emb_op)
    model = fv.to_device(model, device=data.device)

    loss = Loss(obj_tree)
    learn = fv.Learner(data, model, loss_func=loss, **learn_kwargs)
    metrics = BrodenMetrics(learn, obj_tree=obj_tree, preds_func=obj_tree.cs_preds_func, restrict=False)
    learn.callbacks.extend([metrics, utils.AddTargetClbk()])

    learn.split([learn.model.td[0]])
    if pretrained:
        learn.freeze()
    return learn


def upernet_data_pipeline(broden_root, norm_stats=None):
    adapter_tfm = utils.UperNetAdapter()
    train_collate = utils.ScaleJitterCollate([384, 480, 544, 608, 672])
    val_collate = utils.ScaleJitterCollate([544])
    db = get_data(broden_root, size=None, norm_stats=norm_stats,
                  max_rotate=None, max_zoom=1, max_warp=None, max_lighting=None,
                  bs=8, no_check=True, dl_tfms=adapter_tfm)
    db.train_dl.dl.collate_fn = train_collate
    db.valid_dl.dl.collate_fn = val_collate
    return db
