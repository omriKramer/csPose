from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fastai.vision import ItemList, Tensor, Any, ImagePoints, FlowField, scale_flow, LearnerCallback, add_metrics, \
    ImageList, tensor, TfmPixel
from fastai.vision import PreProcessor

from eval import heatmap_to_preds
from lip_utils import plot_joint


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

    def __call__(self, o):
        phase = o.parent.name.partition('_')[0]
        df = self.train_df if phase == 'train' else self.val_df
        pose = df.loc[o.name].values
        pose = torch.tensor(pose, dtype=torch.float).reshape(-1, 3)
        pose = torch.index_select(pose, 1, torch.tensor([1, 0, 2], dtype=torch.long))
        return pose


def output_to_scaled_pred(output):
    if output.ndim == 2:
        # flattened regression
        return output.reshape(output.shape[0], -1, 2)
    elif output.ndim == 4:
        # heatmaps
        h, w = output.shape[-2:]
        pred = heatmap_to_preds(output, add_visibility=False).flip(-1)
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
             **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        data = self.data
        pnt = data[:, :2]
        visible = data[:, 2]
        pnt = scale_flow(FlowField(self.size, pnt), to_unit=False).flow.flip(1)
        plot_joint(ax, pnt, visible)
        if hide_axis:
            ax.axis('off')
        if title:
            ax.set_title(title)

    def flip_lr(self):
        self.flow.flow[..., 0] *= -1
        self.flow.flow = self.flow.flow[self.switch_on_lr_flip]
        self.visible = self.visible[self.switch_on_lr_flip]
        return self


class PoseProcessor(PreProcessor):

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
        pred = output_to_scaled_pred(pred[None])[0]
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


class PoseLoss:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, output, target):
        if output.ndim == 2:
            output = output.reshape(-1, 16, 2)
        is_visible = target[..., 2] > 0
        gt = target[..., :2][is_visible]
        output = output[is_visible]
        return self.loss_fn(output, gt)


class Pckh(LearnerCallback):
    _order = -20  # Needs to run before the recorder

    def __init__(self, learn):
        super().__init__(learn)
        self.all_idx = list(range(0, 6)) + list(range(8, 16))

    def on_train_begin(self, **kwargs: Any) -> None:
        metrics = ['Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'UBody', 'Total']
        self.learn.recorder.add_metric_names(metrics)

    def on_epoch_begin(self, **kwargs: Any) -> None:
        self.correct = torch.zeros(16)
        self.total = torch.zeros(16)

        self.ubody_correct = 0.
        self.ubody_total = 0.

        self.all_keypoints_correct = 0.
        self.all_keypoints_total = 0.

    def on_batch_end(self, last_output, last_target, train, **kwargs) -> None:
        if train:
            return

        preds = output_to_scaled_pred(last_output)
        is_visible = last_target[..., 2] > 0
        gt = last_target[..., :2]

        # remove image without head segment
        has_head = (is_visible[:, 8:10]).all(1)
        preds = preds[has_head]
        gt = gt[has_head]
        is_visible = is_visible[has_head]

        head_sizes = torch.norm(gt[:, 8] - gt[:, 9], dim=1)
        thresholds = head_sizes / 2
        distances = torch.norm(preds - gt, dim=2)
        is_correct = (distances < thresholds[:, None]) * is_visible

        self.correct += is_correct.sum(dim=0)
        self.total += is_visible.sum(dim=0)
        self.ubody_correct += is_correct[:, 8:].sum().item()
        self.ubody_total += is_visible[:, 8:].sum().item()
        self.all_keypoints_correct += is_correct[:, self.all_idx].sum().item()
        self.all_keypoints_total += is_visible[:, self.all_idx].sum().item()

    def on_epoch_end(self, last_metrics, **kwargs):
        idx_pairs = [(8, 9), (12, 13), (11, 14), (10, 15), (2, 3), (1, 4), (0, 5)]
        accuracy = self.correct / self.total
        pckh = [(accuracy[idx0] + accuracy[idx1]).item() / 2
                for idx0, idx1
                in idx_pairs]
        pckh.extend([self.ubody_correct / self.ubody_total, self.all_keypoints_correct / self.all_keypoints_total])
        return add_metrics(last_metrics, pckh)


def _pose_flip_lr(x):
    "Flip `x` horizontally."
    if isinstance(x, Pose):
        return x.flip_lr()
    return tensor(np.ascontiguousarray(np.array(x)[..., ::-1]))


pose_flip_lr = TfmPixel(_pose_flip_lr)
