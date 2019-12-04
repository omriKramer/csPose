import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import csmodels
import engine as eng
import transform
import utils
from datasets import CocoSingleKPS

DUMMY_INSTRUCTION = 0
IMAGE_SIZE = 256, 256


def extract_keypoints(image, target):
    target = target['keypoints'][:2]
    return image, target


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
        heatmap = outputs.squeeze(dim=1)
        targets = targets.round()
        loss = ce_loss(heatmap, targets.long())
        with torch.no_grad():
            preds = heatmap_to_preds(heatmap)
            distances = F.pairwise_distance(preds.to(dtype=torch.float32), targets)
        return {
            'loss': loss,
            'L2': distances,
        }


def ce_loss(heatmap, targets):
    h, w = heatmap.shape[-2:]
    heatmap = heatmap.flatten(start_dim=-2)
    assert not torch.isnan(heatmap).any(), 'Output of model has nans'
    targets = targets[:, 1] * w + targets[:, 0]
    loss = F.cross_entropy(heatmap, targets)
    return loss


mean = np.array([0.4064, 0.3758, 0.3585])
std = np.array([0.2377, 0.2263, 0.2234])


def show_image(ax, image):
    image = std * image + mean
    image = image.clip(0, 1)
    ax.imshow(image)


def plot(batch_results, images, targets, outputs):
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    preds = heatmap_to_preds(outputs[:, 0])
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    fig, axis = plt.subplots(1, min(len(images), 4))
    fig.set_tight_layout(True)
    for ax, distance, image, t, p in zip(axis, batch_results['L2'], images, targets, preds):
        show_image(ax, image)
        ax.plot(t[0], t[1], 'ro', label='target')
        ax.plot(p[0], p[1], 'bo', label='prediction')
        ax.set_title(f'L2: {distance:.2f}')
        ax.set_axis_off()

    handles, labels = axis[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    return 'predictions vs. actuals', fig


if __name__ == '__main__':
    data_path, remaining_args = utils.get_data_path()
    engine = eng.Engine.command_line_init(args=remaining_args)

    data_transform = transform.Compose([
        transform.ResizeKPS(IMAGE_SIZE),
        extract_keypoints,
        transform.ToTensor(),
        transform.ImageTargetWrapper(T.Normalize(mean, std))
    ])

    keypoints = 'left_eye'
    coco_train = CocoSingleKPS.from_data_path(data_path, train=True, transforms=data_transform, keypoints=keypoints)
    coco_val = CocoSingleKPS.from_data_path(data_path, train=False, transforms=data_transform, keypoints=keypoints)

    model = csmodels.resnet50(layers_out=1, num_instructions=1)
    model.one_iteration()
    instructions = [DUMMY_INSTRUCTION]
    model = csmodels.SequentialInstructor(model, instructions)

    train_eval = Evaluator()
    val_eval = Evaluator()
    engine.run(model, coco_train, coco_val, train_eval, val_eval, plot_fn=plot)
