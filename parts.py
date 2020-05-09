from collections import OrderedDict
from pathlib import Path

import fastai.vision as fv
import torch
import torch.nn.functional as F
from torch import nn


class ObjectAndParts(fv.ItemBase):

    def __init__(self, objects: fv.ImageSegment, parts: fv.ImageSegment):
        assert objects.shape == parts.shape
        self.objects = objects
        self.parts = parts

    @property
    def data(self):
        return self.objects.data.squeeze(), self.parts.data.squeeze()

    def apply_tfms(self, tfms, **kwargs):
        objects = self.objects.apply_tfms(tfms, **kwargs)
        parts = self.parts.apply_tfms(tfms, **kwargs)
        return self.__class__(objects, parts)

    def __repr__(self):
        return f'{self.__class__.__name__} {tuple(self.objects.size)}'


class ObjectsPartsLabelList(fv.ItemList):

    def __init__(self, items, **kwargs):
        super().__init__(items, **kwargs)

    def get(self, i):
        object_fn, parts_fn = super().get(i)
        obj = fv.open_mask(object_fn, convert_mode='I')
        if parts_fn:
            parts = fv.open_mask(parts_fn, convert_mode='L')
        else:
            parts = fv.ImageSegment(torch.zeros_like(obj.px))
        return ObjectAndParts(obj, parts)

    def analyze_pred(self, pred):
        raise NotImplemented

    def reconstruct(self, t, x=None):
        obj = fv.ImageSegment(t[0])
        parts = fv.ImageSegment(t[1])
        return ObjectAndParts(obj, parts)


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


def pix_acc(pred, gt):
    mask = gt > 0
    correct = (pred == gt) * mask
    correct = correct.sum()
    total = mask.sum()
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


class Accuracy:
    def __init__(self, n=1):
        self.correct = torch.zeros(n)
        self.total = torch.zeros(n)

    def update(self, correct, total):
        self.correct += correct.cpu()
        self.total += total.cpu()

    def accuracy(self):
        c = self.correct.float()
        t = self.total.float()
        return torch.mean(c / t).item()


class ObjectTree:

    def __init__(self, tree, obj_names):
        self.tree = OrderedDict(sorted(tree.items(), key=lambda item: item[0]))
        self.obj2idx = {o: i for i, o in enumerate(self.tree.keys())}
        self.obj_names = obj_names

    def obj_and_parts(self):
        return self.tree.items()

    @property
    def n_obj(self):
        return len(self.obj_names)

    @property
    def obj_with_parts(self):
        return list(self.tree.keys())

    @property
    def n_obj_with_parts(self):
        return len(self.obj_with_parts)

    @property
    def n_parts(self):
        return sum(len(parts) for parts in self.tree.values())

    def split_parts_pred(self, t: fv.Tensor):
        """
        Split parts prediction by objects.
        Args:
            t: Tensor of shape: (bs, n_parts, h , w)

        Returns: List of length n_obj_with_parts where each item is tensor of shape (bs, n_parts_i, h, w)

        """
        return t.split(self.sections, dim=1)

    def split_parts_gt(self, obj: fv.Tensor, part: fv.Tensor):
        """
        Splits parts gt by objects.
        Args:
            obj: Tensor of shape: (bs, h , w)
            part: Tensor of shape: (bs, h , w)

        Returns: Tensor of shape (n_obj_with_parts, bs, h, w)

        """

        present_obj = obj.unique().cpu().tolist()
        present_obj = [o for o in present_obj if o in self.tree]
        classes = torch.tensor(self.obj_with_parts, device=obj.device)
        obj_masks = obj == classes[:, None, None, None]
        parts_inside_obj = part[None] * obj_masks
        gt = torch.full_like(parts_inside_obj, -1)
        for o in present_obj:
            obj_parts = self.tree[o]
            i = self.obj2idx[o]
            inside_obj_i = parts_inside_obj[i]
            for part_idx, part in enumerate(obj_parts[1:], start=1):
                part_mask = inside_obj_i == part
                gt[i][part_mask] = part_idx

        # if an object has parts then label background (non-part) pixels inside the object with 0
        is_part = gt > 0
        has_parts = is_part.flatten(start_dim=2).any(dim=2, keepdim=True)[..., None]
        bg_inside_obj = has_parts * obj_masks * (~is_part)
        gt[bg_inside_obj] = 0
        return gt

    @property
    def sections(self):
        sections = [len(parts) for parts in self.tree.values()]
        return sections


class BrodenMetrics(fv.LearnerCallback):
    _order = -20

    def __init__(self, learn, obj_tree: ObjectTree, preds_func=None):
        super().__init__(learn)
        self.obj_tree = obj_tree
        self.preds_func = preds_func
        self._reset()

    def _reset(self):
        self.obj_pa = Accuracy()
        self.obj_iou = Accuracy(self.obj_tree.n_obj - 1)

        self.part_pa = Accuracy()
        self.part_iou = [Accuracy(n - 1) for n in self.obj_tree.sections]

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['object-P.A.', 'object-mIoU', 'part-P.A.', 'part-mIoU(bg)'])

    def on_epoch_begin(self, **kwargs):
        self._reset()

    def on_batch_end(self, last_output, last_target, train, **kwargs):
        if train:
            return

        obj_gt, part_gt = last_target
        part_gt = self.object_tree.split_parts_gt(obj_gt, part_gt)

        if self.preds_func:
            last_output = self.preds_func(last_output)
        obj_pred, part_pred = last_output

        obj_pred = obj_pred.argmax(dim=1)
        part_pred = self.obj_tree.split_parts_pred(part_pred)
        # part_pred shape: (n_obj_with_parts, bs, h, w)
        part_pred = torch.stack([obj_parts.argmax(dim=1) for obj_parts in part_pred], dim=0)
        objects_with_parts = torch.tensor(self.obj_tree.obj_with_parts, device=obj_pred.device)[:, None, None, None]
        object_pred_mask = obj_pred == objects_with_parts
        part_pred = part_pred * object_pred_mask

        obj_pred, part_pred = resize_obj_part(obj_pred, part_pred, obj_gt.shape[-2:])

        self.obj_pa.update(*pix_acc(obj_pred, obj_gt))
        self.obj_iou.update(*iou(obj_pred, obj_gt, self.obj_tree.n_obj, obj_gt > 0))

        self.part_pa.update(*pix_acc(part_pred, part_gt))
        for i, (obj, parts) in enumerate(self.obj_tree.obj_and_parts()):
            mask = part_gt[i] > -1
            self.part_iou[i].update(*iou(part_pred[i], part_gt[i], len(parts), mask))

    def on_epoch_end(self, last_metrics, **kwargs):
        parts_iou = [c.accuracy() for c in self.part_iou]
        parts_iou = sum(parts_iou) / len(parts_iou)
        results = [
            self.obj_pa.accuracy(),
            self.obj_iou.accuracy(),
            self.part_pa.accuracy(),
            parts_iou
        ]

        return fv.add_metrics(last_metrics, results)


def resize_obj_part(obj, part, size):
    if obj.shape[-2:] != size:
        obj = F.interpolate(obj[None].float(), size=size, mode='nearest')[0].long()
        part = F.interpolate(part.float(), size=size, mode='nearest').long()
    return obj, part


class Loss:

    def __init__(self, object_tree: ObjectTree, preds_func=None):
        self.object_tree = object_tree
        # unlabeled pixels are zeros
        self.obj_ce = nn.CrossEntropyLoss(ignore_index=0)
        # outside objects gt values are expected to be -1, background inside objects is 0
        self.part_ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.preds_func = preds_func

    def __call__(self, pred, obj_gt, part_gt):
        if self.preds_func:
            pred = self.preds_func(pred)

        obj_pred, part_pred = pred
        obj_gt, part_gt = resize_obj_part(obj_gt, part_gt, obj_pred.shape[-2:])
        part_gt = self.object_tree.split_parts_gt(obj_gt, part_gt)

        obj_loss = self.obj_ce(obj_pred, obj_gt)
        part_loss = []
        for i, obj_i_parts in enumerate(self.object_tree.split_parts_pred(part_pred)):
            if torch.any(part_gt[i] > -1):
                part_loss.append(self.part_ce(obj_i_parts, part_gt[i]))

        loss = obj_loss + sum(part_loss)
        return loss


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


def get_data(broden_root, size=256, bs=8, norm_stats=fv.imagenet_stats, padding_mode='zeros'):
    labeler = Labeler(broden_root / 'reindexed2')
    tfms = fv.get_transforms()

    data = (ObjectsPartsItemList.from_csv(broden_root, 'trainval.csv')
            .split_from_df(col='is_valid')
            .label_from_func(labeler)
            .transform(tfms, tfm_y=True, size=size, resize_method=fv.ResizeMethod.PAD, padding_mode=padding_mode)
            .databunch(bs)
            .normalize(norm_stats))
    return data
