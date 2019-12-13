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


class Evaluator:

    def __call__(self, outputs, targets):
        """
        outputs: (N, K, H, W)
        targets: (N, K, 2)
        """
        targets = targets.round()
        loss = ce_loss(outputs, targets.long())
        with torch.no_grad():
            distances = pairwise_distance(outputs, targets)
        return {
            'loss': loss,
            'mean_distance': distances,
        }


def pairwise_distance(outputs, targets):
    preds = heatmap_to_preds(outputs).to(dtype=torch.float32)
    distances = [F.pairwise_distance(p, t).mean() for p, t in zip(preds, targets)]
    distances = torch.stack(distances)
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

    def __call__(self, batch_results, images, targets, outputs):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        preds = heatmap_to_preds(outputs).cpu().numpy()
        targets = targets.cpu().numpy()

        fig, axis = plt.subplots(1, min(len(images), 4))
        fig.set_tight_layout(True)
        for ax, distance, image, img_t, img_p in zip(axis, batch_results['mean_distance'], images, targets, preds):
            show_image(ax, image, self.mean, self.std)
            ax.plot(img_t[:, 0], img_t[:, 1], 'or', label='target')
            ax.plot(img_p[:, 0], img_p[:, 1], 'ob', label='prediction')
            ax.set_title(f'mean pairwise distance: {distance:.2f}')
            ax.set_axis_off()

        handles, labels = axis[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        return 'predictions vs. actuals', fig
