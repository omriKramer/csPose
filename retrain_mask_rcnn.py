import torch
import torchvision

import coco_eval
import coco_utils
import engine.engine as eng
import transform


class WrapInList:

    def __call__(self, image, target):
        return image, [target]

    def __repr__(self):
        return f'{self.__class__.__name__}()'


cpu_device = torch.device('cpu')


def target_to_coco_format(list_of_dict):
    batch = {'area': [], 'keypoints': []}
    for d in list_of_dict:
        batch['keypoints'].append(d['keypoints'][0].reshape(-1).to(cpu_device))
        batch['area'].append(d['area'].to(cpu_device))

    return batch


def output_to_single_kps(list_of_dict):
    batch = []
    for d in list_of_dict:
        keypoints = d['keypoints']
        if keypoints.nelement() > 0:
            keypoints = keypoints[0].reshape(-1)

        batch.append(keypoints.to(cpu_device))

    return batch


if __name__ == '__main__':
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, min_size=80, max_size=640)

    engine = eng.Engine.command_line_init(model, optimizer=torch.optim.Adam, model_feeder=eng.feed_images_and_targets)
    train_transform = transform.Compose(
        (WrapInList(), transform.ConvertCocoPolysToMask(), transform.ToTensor(), transform.RandomHorizontalFlip(0.5)))

    val_transform = transform.Compose((WrapInList(), transform.ConvertCocoPolysToMask(), transform.ToTensor()))
    train_metrics = {
        'loss': lambda losses_dict, _: sum(loss for loss in losses_dict.values()),
    }
    val_metrics = {
        'OKS': lambda outputs, target: coco_eval.batch_oks(output_to_single_kps(outputs),
                                                           target_to_coco_format(target)),
    }
    coco_train = engine.get_dataset(train=True, transforms=train_transform)
    coco_val = engine.get_dataset(train=False, transforms=val_transform)
    engine.run(coco_train, coco_val, train_metrics, val_metrics=val_metrics, collate_fn=coco_utils.collate_fn)
