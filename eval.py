import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

import models.layers


def heatmap_to_preds(heatmap):
    h, w = heatmap.shape[-2:]
    heatmap = heatmap.flatten(start_dim=-2)
    preds = heatmap.argmax(dim=-1)
    y = preds // w
    x = preds.remainder(w)
    visible = torch.ones_like(x)
    preds = torch.stack((x, y, visible), dim=-1)
    return preds


def resize_kps(kps, new_size, original_size):
    scale_x = new_size[1] / original_size[1]
    scale_y = new_size[0] / original_size[0]
    new_x = kps[..., 0] * scale_x
    new_y = kps[..., 1] * scale_y
    new_x = new_x.clamp(0, new_size[1] - 1)
    new_y = new_y.clamp(0, new_size[0] - 1)
    new_kps = torch.stack((new_x, new_y, kps[..., 2]), dim=-1)
    inds = new_kps[..., 2] == 0
    new_kps[inds] = 0
    return new_kps


class Evaluator(nn.Module):

    def __init__(self, original_size=None, loss='ce'):
        super(Evaluator, self).__init__()
        self.original_size = original_size
        if loss == 'ce':
            self.loss = ce_loss
        elif loss == 'kl':
            self.loss = KLLoss()
        else:
            raise ValueError(f'Got loss {loss}, expected ce or kl.')

    def forward(self, outputs, targets):
        """
        outputs: (N, K, H, W)
        targets: (N, K, 2)
        """
        loss_targets = targets
        if self.original_size:
            loss_targets = resize_kps(targets, outputs.shape[-2:], self.original_size)
        loss = self.loss(outputs, loss_targets)

        with torch.no_grad():
            preds = heatmap_to_preds(outputs).float()
            if self.original_size:
                preds = resize_kps(preds, self.original_size, outputs.shape[-2:])
            mean_distance = pairwise_distance(preds, targets)
        meters = {'loss': loss, 'mean_distance': mean_distance}
        return meters, preds


def pairwise_distance(preds, targets):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    visible = targets[..., 2] > 0
    preds = preds[..., :2]
    targets = targets[..., :2]

    distances = np.array([np.linalg.norm(p - t, axis=1) for p, t in zip(preds, targets)])
    distances[~visible] = None
    mean_distance = np.nanmean(distances, axis=1)
    assert np.all(~np.isnan(mean_distance))
    return mean_distance


def mse(heatmap, targets):
    h, w = heatmap.shape[-2:]
    targets = targets[:, 1] * w + targets[:, 0]
    targets = F.one_hot(targets, num_classes=h * w).reshape(-1, h, w)
    return F.mse_loss(heatmap, targets.to(dtype=torch.float))


def ce_loss(heatmap, targets):
    assert not torch.isnan(heatmap).any(), 'Output of model has nans'
    n, k, h, w = heatmap.shape
    heatmap = heatmap.view(n * k, h * w)
    targets = targets.view(n * k, 3)
    targets = targets.round().long()
    visible = targets[..., 2] > 0
    targets = targets[..., 1] * w + targets[..., 0]
    loss = F.cross_entropy(heatmap[visible], targets[visible])
    return loss


def one_hot2d(x, h, w):
    out = x[..., 1] * w + x[..., 0]
    out = F.one_hot(out, h * w)
    out = out.reshape(*x.shape[:-1], h, w)
    return out


class KLLoss(nn.Module):

    def __init__(self):
        super(KLLoss, self).__init__()
        self.smooth = models.layers.GaussianSmoothing(0.5)
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(dim=2)

    def __call__(self, heatmap, targets):
        heatmap = self.log_softmax(heatmap.flatten(start_dim=-2)).reshape_as(heatmap)
        targets = targets.round().long()
        visible = targets[..., 2] > 0
        targets = one_hot2d(targets[..., :2], heatmap.shape[-2], heatmap.shape[-1])
        targets = self.smooth(targets.float())
        loss = self.kl_div_loss(heatmap[visible], targets[visible])
        return loss


class Visualizer:
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = 0
        if std is None:
            std = 1

        self.std = std
        self.mean = mean

    def _unnormalize(self, image):
        image = self.std * image + self.mean
        image = image.clip(0, 1)
        return image

    def __call__(self, batch_results, targets, preds, images):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        fig, axis = plt.subplots(1, min(len(images), 4))
        for ax, distance, image, img_t, img_p in zip(axis, batch_results['mean_distance'], images, targets, preds):
            image = self._unnormalize(image)
            ax.imshow(image)
            ms = 2.5
            ax.plot(img_t[:, 0], img_t[:, 1], 'or', label='target', markersize=ms)
            ax.plot(img_p[:, 0], img_p[:, 1], 'ob', label='prediction', markersize=ms)
            ax.set_title(f'mPD: {distance:.2f}')
            ax.set_axis_off()

        handles, labels = axis[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        return 'predictions vs. actuals', fig
