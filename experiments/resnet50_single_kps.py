import torch
import torchvision

import coco_utils
import engine.engine as eng
import transform
from coco_eval import batch_oks

if __name__ == '__main__':
    model = torchvision.models.resnet50(progress=False, num_classes=3 * len(coco_utils.KEYPOINTS))
    engine = eng.Engine.command_line_init(model, optimizer=torch.optim.Adam)

    composed = transform.Compose([transform.ResizeKPS((80, 150)), transform.ToTensor(keys=('bbox', 'keypoints'))])
    coco_train = engine.get_dataset(train=True, transforms=composed)
    coco_val = engine.get_dataset(train=False, transforms=composed)

    mse_loss = torch.nn.MSELoss()
    metrics = {
        'loss': lambda output, targets: mse_loss(output, targets['keypoints']),
        'OKS': batch_oks,
    }

    engine.run(coco_train, coco_val, metrics)
