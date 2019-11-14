import matplotlib.pyplot as plt
import torch

import engine as eng
import transform
import utils
from datasets import CocoSingleKPS
from engine.eval import MetricLogger
from models import resnet

DUMMY_INSTRUCTION = 0
IMAGE_SIZE = 256, 256


def extract_keypoints(image, target):
    target = target['keypoints'][:2]
    return image, target


def model_feeder(model, images, _):
    return model(images, torch.LongTensor([DUMMY_INSTRUCTION]))


def heatmap_to_preds(heatmap):
    h, w = heatmap.shape[-2:]
    heatmap = heatmap.flatten(start_dim=-2)
    preds = heatmap.argmax(dim=-1)
    y = preds // w
    x = preds.remainder(w)
    preds = torch.stack((x, y), dim=len(x.shape))
    return preds


l2 = torch.nn.PairwiseDistance()


def metrics(targets, outputs):
    preds = heatmap_to_preds(outputs['td'][0])
    distances = l2(preds.to(dtype=torch.float32), targets)
    return {'L2': distances}


ce = torch.nn.CrossEntropyLoss()


def loss(outputs, targets):
    heatmap = outputs['td'][0]
    h, w = heatmap.shape[-2:]
    heatmap = heatmap.flatten(start_dim=-2)
    targets = targets[:, 0] * w + targets[:, 1]
    targets = targets.round().long()
    return ce(heatmap, targets)


def plot(batch_results, images, targets, outputs):
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    preds = heatmap_to_preds(outputs['td'][0])
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    fig, axis = plt.subplots(1, len(images))
    for ax, distance, image, t, p in zip(axis, batch_results['L2'], images, targets, preds):
        ax.imshow(image)
        ax.plot(t[0], t[1], 'ro', label='target')
        ax.plot(p[0], p[1], 'bo', label='prediction')
        ax.set_title(f'L2: {distance:.2f}')

    handles, labels = axis[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    return 'predictions vs. actuals', fig


if __name__ == '__main__':
    data_path, remaining_args = utils.get_data_path()
    train_transform = transform.Compose([transform.ResizeKPS(IMAGE_SIZE), extract_keypoints, transform.ToTensor()])
    val_transform = transform.Compose([transform.ResizeKPS(IMAGE_SIZE), extract_keypoints, transform.ToTensor()])

    keypoints = 'left_eye'
    coco_train = CocoSingleKPS.from_data_path(data_path, train=True, transforms=train_transform, keypoints=keypoints)
    coco_val = CocoSingleKPS.from_data_path(data_path, train=False, transforms=val_transform, keypoints=keypoints)

    resnet18 = resnet.resnet18(layers_out=1, num_instructions=1)
    resnet18.one_iteration()
    engine = eng.Engine.command_line_init(args=remaining_args)

    train_eval = MetricLogger(metrics)
    val_eval = MetricLogger(metrics, plot_fn=plot)
    engine.run(resnet18, torch.optim.Adam, coco_train, coco_val, train_eval, val_eval, loss, model_feeder=model_feeder)
