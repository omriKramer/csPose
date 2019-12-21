import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F


def heatmap_to_preds(heatmap):
    h, w = heatmap.shape[-2:]
    heatmap = heatmap.flatten(start_dim=-2)
    preds = heatmap.argmax(dim=-1)
    y = preds // w
    x = preds.remainder(w)
    preds = torch.stack((x, y), dim=-1)
    return preds


def resize_kps(kps, new_size, original_size):
    scale_x = new_size[1] / original_size[1]
    scale_y = new_size[0] / original_size[0]
    new_x = kps[..., 0] * scale_x
    new_y = kps[..., 1] * scale_y
    new_x = new_x.clamp(0, new_size[1] - 1)
    new_y = new_y.clamp(0, new_size[0] - 1)
    new_kps = torch.stack((new_x, new_y), dim=-1)
    return new_kps


class Evaluator:

    def __init__(self, original_size=None):
        self.original_size = original_size

    def __call__(self, outputs, targets):
        """
        outputs: (N, K, H, W)
        targets: (N, K, 2)
        """
        ce_targets = targets
        if self.original_size:
            ce_targets = resize_kps(targets, outputs.shape[-2:], self.original_size)
        ce_targets = ce_targets.round().long()
        loss = ce_loss(outputs, ce_targets)

        with torch.no_grad():
            preds = heatmap_to_preds(outputs).float()
            if self.original_size:
                preds = resize_kps(preds, self.original_size, outputs.shape[-2:])
            distances = pairwise_distance(preds, targets)
        meters = {
            'loss': loss,
            'mean_distance': distances,
        }
        return meters, preds


def pairwise_distance(preds, targets):
    distances = [F.pairwise_distance(p, t) for p, t in zip(preds, targets)]
    distances = torch.stack(distances)
    distances = distances.mean(dim=1)
    return distances


def mse(heatmap, targets):
    h, w = heatmap.shape[-2:]
    targets = targets[:, 1] * w + targets[:, 0]
    targets = F.one_hot(targets, num_classes=h * w).reshape(-1, h, w)
    return F.mse_loss(heatmap, targets.to(dtype=torch.float))


def ce_loss(heatmap, targets):
    assert not torch.isnan(heatmap).any(), 'Output of model has nans'
    h, w = heatmap.shape[-2:]
    heatmap = heatmap.flatten(start_dim=-2)
    heatmap = heatmap.permute(0, 2, 1)
    targets = targets[..., 1] * w + targets[..., 0]
    loss = F.cross_entropy(heatmap, targets)
    return loss


def show_image(ax, image, mean, std):
    image = std * image + mean
    image = image.clip(0, 1)
    ax.imshow(image)


class Visualizer:
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean

    def __call__(self, batch_results, images, targets, preds):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        fig, axis = plt.subplots(1, min(len(images), 4))
        for ax, distance, image, img_t, img_p in zip(axis, batch_results['mean_distance'], images, targets, preds):
            show_image(ax, image, self.mean, self.std)
            ax.plot(img_t[:, 0], img_t[:, 1], 'or', label='target')
            ax.plot(img_p[:, 0], img_p[:, 1], 'ob', label='prediction')
            ax.set_title(f'mPD: {distance:.2f}')
            ax.set_axis_off()

        handles, labels = axis[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        return 'predictions vs. actuals', fig
