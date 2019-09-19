import torch
import torchvision

import coco_eval
import coco_utils
import engine.engine as eng
import transform
from datasets import CocoSingleKPS
from engine.eval import MetricLogger


class WrapInList:

    def __call__(self, image, target):
        return image, [target]

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def target_to_coco_format(list_of_dict):
    batch = {'area': [], 'keypoints': []}
    for d in list_of_dict:
        batch['keypoints'].append(d['keypoints'][0].reshape(-1))
        batch['area'].append(d['area'])

    return batch


def output_to_single_kps(list_of_dict):
    batch = []
    for d in list_of_dict:
        keypoints = d['keypoints']
        if keypoints.nelement() > 0:
            keypoints = keypoints[0].reshape(-1)

        batch.append(keypoints)

    return batch


def train_metrics(_, loss_dict):
    return loss_dict


def loss_fn(loss_dict, _):
    return sum(loss for loss in loss_dict.values())


if __name__ == '__main__':
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, min_size=80, max_size=640)
    engine = eng.Engine.command_line_init(model, optimizer=torch.optim.Adam, model_feeder=eng.feed_images_and_targets)

    train_transform = transform.Compose(
        (WrapInList(), transform.ConvertCocoPolysToMask(), transform.ToTensor(), transform.RandomHorizontalFlip(0.5)))
    val_transform = transform.Compose((WrapInList(), transform.ConvertCocoPolysToMask(), transform.ToTensor()))

    coco_train = CocoSingleKPS.from_data_path(engine.data_path, train=True, transforms=train_transform)
    coco_val = CocoSingleKPS.from_data_path(engine.data_path, train=False, transforms=val_transform)

    coco_evaluator = coco_eval.CocoEval(device=engine.device)


    def val_metrics(targets, outputs):
        return {
            'OKS': coco_evaluator.batch_oks(output_to_single_kps(outputs), target_to_coco_format(targets)).to(
                engine.device),
        }


    train_evaluator = MetricLogger(train_metrics)
    val_evaluator = MetricLogger(val_metrics)

    engine.run(coco_train, coco_val, train_evaluator, val_evaluator, loss_fn, collate_fn=coco_utils.collate_fn)
