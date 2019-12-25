import json

import torch
import torchvision

import datasets
import engine
import transform
import utils


def collate_fn(batch):
    return tuple(zip(*batch))


class Logger:

    def __init__(self, file):
        self.file = file
        self.results = {}

    def __call__(self, outputs, targets):
        for image_preds, image_targets in zip(outputs, targets):
            self.results[image_targets['image_id'].item()] = {k: v.tolist() for k, v in image_preds.items()}

            return {}, None

    def dump(self):
        with open(self.file, 'w') as f:
            json.dump(self.results, f)


def main():
    root, args = utils.get_data_path()
    ds = datasets.get_coco_kp(root, 'val', transform.ToTensor())
    data_loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    eng = engine.Engine.command_line_init(args)
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.to(eng.device)
    logger = Logger(eng.output_dir / 'keypoints_rcnn_val.json')
    eng.evaluate(model, data_loader, logger, 0)
    logger.dump()


if __name__ == '__main__':
    main()
