from typing import Optional, Tuple

import fastai.vision as fv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastai.vision import ItemList, Tensor, Any, ImagePoints, FlowField, scale_flow, LearnerCallback, add_metrics, \
    ImageList, tensor, TfmPixel

import lip_utils
from eval import heatmap_to_preds


def _get_size(xs, i):
    size = xs.sizes.get(i, None)
    if size is None:
        # Image hasn't been accessed yet, so we don't know its size
        _ = xs[i]
        size = xs.sizes[i]
    return size


def _mark_points_out(flow, visible):
    pad_mask = (flow.flow[:, 0] >= -1) * (flow.flow[:, 0] <= 1) * (flow.flow[:, 1] >= -1) * (flow.flow[:, 1] <= 1)
    visible = visible.where(pad_mask, torch.zeros(1))
    return visible


class LIPLabel:

    def __init__(self, ann_folder):
        self.train_df = pd.read_csv(ann_folder / 'lip_train_set.csv', index_col=0, header=None)
        self.val_df = pd.read_csv(ann_folder / 'lip_val_set.csv', index_col=0, header=None)
        self.no_ann = pd.read_csv(ann_folder / 'train_no_pose.txt', header=None, index_col=0).index

    def __call__(self, o):
        phase = o.parent.name.partition('_')[0]
        df = self.train_df if phase == 'train' else self.val_df
        pose = df.loc[o.name].values
        pose = torch.tensor(pose, dtype=torch.float).reshape(-1, 3)
        pose = torch.index_select(pose, 1, torch.tensor([1, 0, 2], dtype=torch.long))
        return pose

    def filter(self, o):
        if o.name in self.no_ann:
            return False

        return True


def output_to_scaled_pred(output):
    h, w = output.shape[-2:]
    pred = heatmap_to_preds(output, add_visibility=False).flip(-1).float()
    s = pred.new([h / 2, w / 2])[None, None]
    pred = pred / s - 1
    return pred


class Pose(ImagePoints):
    switch_on_lr_flip = list(reversed(range(6))) + list(range(6, 10)) + list(reversed(range(10, 16)))

    def __init__(self, flow: FlowField, visible, scale: bool = True, y_first: bool = True, mode='LIP'):
        super().__init__(flow, scale, y_first)
        if mode == 'LIP':
            visible = torch.where(torch.isnan(visible), torch.zeros(1), visible + 1)
        self.visible = visible
        self._flow.flow.clamp_(-1, 1)

    def clone(self):
        cloned_flow = FlowField(self.size, self.flow.flow.clone())
        return self.__class__(cloned_flow, self.visible.clone(), scale=False, y_first=False, mode='COCO')

    @property
    def n_vis(self):
        return torch.sum(self.visible > 0).item()

    @property
    def n_points(self):
        return len(self.visible)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.n_vis}/{self.n_points}) {tuple(self.size)}'

    @property
    def data(self) -> Tensor:
        flow = self.flow  # This updates flow before we test if some transforms happened
        visible = self.visible
        if self.transformed:
            if 'remove_out' not in self.sample_kwargs or self.sample_kwargs['remove_out']:
                visible = _mark_points_out(flow, self.visible)
            self.transformed = False
        pose = flow.flow.flip(1)
        return torch.cat((pose, visible[:, None]), dim=1)

    def show(self, ax: plt.Axes = None, figsize: tuple = (3, 3), title: Optional[str] = None, hide_axis: bool = True,
             annotate=False, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        data = self.data
        pnt = data[:, :2]
        visible = data[:, 2]
        pnt = scale_flow(FlowField(self.size, pnt), to_unit=False).flow.flip(1)
        lip_utils.plot_joint(ax, pnt, visible, annotate=annotate)
        if hide_axis:
            ax.axis('off')
        if title:
            ax.set_title(title)

    def flip_lr(self):
        self.flow.flow[..., 0] *= -1
        self.flow.flow = self.flow.flow[self.switch_on_lr_flip]
        self.visible = self.visible[self.switch_on_lr_flip]
        return self


class PoseProcessor(fv.PreProcessor):

    def __init__(self, ds: ItemList):
        super().__init__(ds)
        self.c = len(ds.items[0][:, :2].reshape(-1))

    def process(self, ds: ItemList):  ds.c = self.c


class PoseLabelList(ItemList):
    _processor = PoseProcessor

    def __init__(self, items, **kwargs):
        super().__init__(items, **kwargs)

    def get(self, i) -> Any:
        o = super().get(i)
        flow = FlowField(_get_size(self.x, i), o[:, :2])
        return Pose(flow, o[:, 2], scale=True)

    def reconstruct(self, t: Tensor, x: Tensor = None):
        flow = FlowField(x.size, t[:, :2])
        return Pose(flow, t[:, 2], scale=False, mode='COCO')

    def analyze_pred(self, pred: Tensor):
        pred = output_to_scaled_pred(pred[1][None])[0]
        pred.clamp_(-1, 1)
        visibility = pred.new_ones(pred.shape[:-1])
        pred = torch.cat((pred, visibility[..., None]), dim=-1)
        return pred


class PoseItemList(ImageList):
    _label_cls, _square_show_res = PoseLabelList, False

    def show_xyzs(self, xs, ys, zs, imgsize: int = 4, figsize: Optional[Tuple[int, int]] = None, **kwargs):
        for y, z in zip(ys, zs):
            z.visible = y.visible
        super().show_xyzs(xs, ys, zs, imgsize, figsize, **kwargs)


def pose_ce_loss(output, targets):
    is_visible = targets[..., 2] > 0
    gt = targets[..., :2][is_visible]
    output = output[is_visible]
    return ce_loss(output, gt)


def ce_loss(heatmaps, targets):
    h, w = heatmaps.shape[-2:]
    heatmaps = heatmaps.view(-1, h * w)
    targets = scale_targets(targets, (h, w)).round().long()
    # y coordinates are first
    targets = targets[..., 0] * w + targets[..., 1]
    loss = F.cross_entropy(heatmaps, targets)
    return loss


def scale_targets(targets, size):
    rescale = targets.new([size[0] / 2, size[1] / 2])[None]
    targets = (targets + 1) * rescale
    max_size = max(size)
    assert targets.min().item() >= -1
    assert targets.max().item() <= max_size
    targets = targets.clamp(0, max_size - 1)
    return targets


class Pckh(LearnerCallback):
    _order = -20  # Needs to run before the recorder
    all_idx = list(range(0, 6)) + list(range(8, 16))

    def __init__(self, learn, heatmap_func=None, filter_idx=None, acc_thresh=None, niter=1):
        super().__init__(learn)
        if filter_idx and acc_thresh:
            raise ValueError('No support for partial keypoints and multilabel classification')

        self.filter_idx = sorted(filter_idx) if filter_idx else range(16)
        self.heatmap_func = heatmap_func if heatmap_func else lambda outputs: outputs[1]
        self.acc_thresh = acc_thresh
        self.niter = niter

    def on_train_begin(self, **kwargs: Any) -> None:
        metrics = ['Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'UBody', 'Total']
        if self.niter > 1:
            metrics = [f'{title}_{i}' for title in metrics for i in range(self.niter)]
        if self.acc_thresh:
            metrics.extend([f'acc@{self.acc_thresh}', 'TP_acc', 'FN_acc'])
        self.learn.recorder.add_metric_names(metrics)

    def on_epoch_begin(self, **kwargs: Any) -> None:
        self.correct = torch.zeros(self.niter, 18)
        self.total = torch.zeros(self.niter, 18)
        self.mlc_correct = 0
        self.mlc_total = 0

    def on_batch_end(self, last_output, last_target, train, **kwargs) -> None:
        if train:
            return

        preds = output_to_scaled_pred(self.heatmap_func(last_output))
        is_visible = last_target[..., 2] > 0
        gt = last_target[..., :2]

        mlc_pred = None
        if self.acc_thresh:
            mlc_pred = last_output[0][:, -16:].sigmoid() > self.acc_thresh
            self.mlc_correct += (mlc_pred == is_visible.bool()).sum().item()
            self.mlc_total += mlc_pred.numel()

        # remove image without head segment
        has_head = (is_visible[:, 8:10]).all(1)
        preds = preds[has_head]
        gt = gt[has_head]
        is_visible = is_visible[has_head]
        if mlc_pred is not None:
            mlc_pred = mlc_pred[has_head]

        head_sizes = torch.norm(gt[:, 8] - gt[:, 9], dim=1)
        thresholds = (head_sizes / 2)

        # keep only keypoints of interest
        gt = gt[:, self.filter_idx]
        is_visible = is_visible[:, self.filter_idx]

        # update keypoints stats fore each of the models iterations
        for i, p in enumerate(preds.chunk(self.niter, dim=1)):
            distances = torch.norm(p - gt, dim=2)
            is_correct = (distances < thresholds[:, None]) * is_visible
            self.update(is_correct, is_visible, i, mlc_pred)

    def update(self, is_correct, is_visible, i, mlc_pred):
        is_correct = is_correct.cpu().detach()
        is_visible = is_visible.cpu().detach()

        self.correct[i, self.filter_idx] += is_correct.sum(dim=0)
        self.total[i, self.filter_idx] += is_visible.sum(dim=0)

        if mlc_pred is None:
            return

        mlc_pred = mlc_pred.cpu().detach()
        tp = mlc_pred * is_visible
        fn = ~mlc_pred * is_visible
        tp_fn = torch.stack((tp, fn))
        self.correct[i, 16:] += (tp_fn * is_correct[None]).sum(dim=(1, 2))
        self.total[i, 16:] += tp_fn.sum(dim=(1, 2))

    def on_epoch_end(self, last_metrics, **kwargs):
        idx_pairs = [(8, 9), (12, 13), (11, 14), (10, 15), (2, 3), (1, 4), (0, 5)]
        accuracy = self.correct / self.total
        pckh = [(accuracy[:, idx0] + accuracy[:, idx1]).item() / 2
                for idx0, idx1
                in idx_pairs]

        # add upper body and total
        pckh.extend([
            (self.correct[:, 8:].sum(dim=1) / self.total[:, 8:].sum(dim=1)).item(),
            (self.correct[:, self.all_idx].sum(dim=1) / self.total[:, self.all_idx].sum(dim=1)).item()
        ])

        # add multi-label classification accuracy, TP-accuracy, FN-accuracy
        if self.acc_thresh:
            pckh.extend([
                self.mlc_correct / self.mlc_total if self.mlc_total else np.nan,
                accuracy[16].item(),
                accuracy[17].item()
            ])

        results = torch.tensor(pckh)
        results = results.T.view(-1).tolist()
        return add_metrics(last_metrics, results)


def _pose_flip_lr(x):
    """Flip `x` horizontally."""
    if isinstance(x, Pose):
        return x.flip_lr()
    return tensor(np.ascontiguousarray(np.array(x)[..., ::-1]))


pose_flip_lr = TfmPixel(_pose_flip_lr)


def get_data(root, size, bs=64, stats=lip_utils.stats):
    t = fv.get_transforms(do_flip=False)
    t[0].insert(0, pose_flip_lr(p=0.5))
    pose_label = LIPLabel(root / 'pose_annotations')
    data = (PoseItemList.from_folder(root)
            .filter_by_func(pose_label.filter)
            .split_by_folder('train_images', 'val_images')
            .label_from_func(pose_label)
            .transform(t, tfm_y=True, size=size, resize_method=fv.ResizeMethod.PAD, padding_mode='zeros')
            .databunch(bs=bs)
            .normalize(stats))

    data.c = 16
    return data
