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


if __name__ == '__main__':
    data_path, remaining_args = utils.get_data_path()
    train_transform = transform.Compose([transform.ResizeKPS(IMAGE_SIZE), extract_keypoints, transform.ToTensor()])
    val_transform = transform.Compose([transform.ResizeKPS(IMAGE_SIZE), extract_keypoints, transform.ToTensor()])

    keypoints = 'left_eye'
    coco_train = CocoSingleKPS.from_data_path(data_path, train=True, transforms=train_transform, keypoints=keypoints)
    coco_val = CocoSingleKPS.from_data_path(data_path, train=False, transforms=val_transform, keypoints=keypoints)

    resnet18 = resnet.resnet18(layers_out=1, num_instructions=1)
    engine = eng.Engine.command_line_init(args=remaining_args)

    train_eval = MetricLogger(metrics)
    val_eval = MetricLogger(metrics)
    engine.run(resnet18, torch.optim.Adam, coco_train, coco_val, train_eval, val_eval, loss, model_feeder=model_feeder)
