import torch

import coco_eval
import coco_utils
import engine.engine as eng
import transform
from datasets import CocoSingleKPS
from engine.eval import MetricLogger
from models import resnet

lookup_order = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'left_elbow', 'left_wrist',
                'right_shoulder', 'right_elbow', 'right_wrist',
                'left_hip', 'left_knee', 'left_ankle',
                'right_hip', 'right_knee', 'right_ankle']

commands = torch.LongTensor([coco_utils.KEYPOINTS.index(pos) for pos in lookup_order])


def flip_to_coco_order(batch):
    return batch[:, range(len(coco_utils.KEYPOINTS))]


def model_feeder(model, images, _):
    return model(images, commands)


cross_entropy = torch.nn.CrossEntropyLoss()


def loss(outputs, targets):
    outputs = flip_to_coco_order(outputs['td'])
    h, w = outputs[0, 0].shape
    batched = []
    t_batched = []
    for td, kps in zip(outputs, targets['keypoints']):
        x, y, v = coco_utils.decode_keypoints(kps)
        x = x[v > 0]
        y = y[v > 0]
        t_batched.append(torch.stack((x, y), dim=1))
        batched.append(td[v > 0])

    batched = torch.cat(batched)
    t_batched = torch.cat(t_batched)

    batched = batched.reshape((batched.shape[0], -1))
    t_batched = t_batched[:, 0] * w + t_batched[:, 1]
    return cross_entropy(batched, t_batched.long())


def heatmap_to_pred(heatmap):
    n, k, h, w = heatmap.shape
    heatmap = heatmap.reshape((n, k, -1))
    _, preds = heatmap.max(dim=2)
    x = preds // h
    y = preds - preds // h
    v = torch.ones_like(x)
    kps = torch.stack((x, y, v), dim=2).reshape(n, -1)
    return kps.to(dtype=torch.float32)


def plot_kps(batch_results, images, targets, outputs):
    n = min(len(images), 4)
    oks = batch_results['OKS'][:n]
    dt = heatmap_to_pred(outputs['td'])[:n]
    gt = targets['keypoints'][:n]
    images = images[:n]
    coco_utils.plot_kps_comparison(oks, images, dt, gt)


if __name__ == '__main__':
    num_keypoints = len(coco_utils.KEYPOINTS)
    resnet34 = resnet.resnet34(layers_out=1, num_instructions=num_keypoints, )
    engine = eng.Engine.command_line_init(resnet34, optimizer=torch.optim.Adam, model_feeder=model_feeder)

    new_size = 256, 256
    keys = ('image_id', 'keypoints', 'area')
    train_transform = transform.Compose(
        (transform.ResizeKPS(new_size), transform.ToTensor(keys=keys)))
    val_transform = transform.Compose((transform.ResizeKPS(new_size), transform.ToTensor(keys=keys),))

    coco_train = CocoSingleKPS.from_data_path(engine.data_path, train=True, transforms=train_transform)
    coco_val = CocoSingleKPS.from_data_path(engine.data_path, train=False, transforms=val_transform)

    coco_evaluator = coco_eval.CocoEval(device=engine.device)


    def metrics(targets, outputs):
        pred_kps = heatmap_to_pred(outputs['td'])
        return {'OKS': coco_evaluator.batch_oks(pred_kps, targets)}


    train_evaluator = MetricLogger(metrics)
    val_evaluator = MetricLogger(metrics, plot_fn=plot_kps)

    engine.run(coco_train, coco_val, train_evaluator, val_evaluator, loss)
