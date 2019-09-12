import torch
import torchvision

import coco_utils
import engine.engine as eng
import transform

if __name__ == '__main__':
    composed = transform.Compose([transform.ResizeKPS((80, 150)), transform.ToTensor()])
    model = torchvision.models.resnet50(progress=False, num_classes=3 * len(coco_utils.KEYPOINTS))

    engine = eng.Engine.from_command_line(model, optimizer=torch.optim.Adam)
    engine.run(transforms=composed)
