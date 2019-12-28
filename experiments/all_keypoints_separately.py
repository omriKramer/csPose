import torchvision.transforms as T

import coco_utils
import engine as eng
import eval
import models
import transform
import utils
from datasets import CocoSingleKPS

IMAGE_SIZE = 128, 128

data_path, remaining_args = utils.get_data_path()
engine = eng.Engine.command_line_init(args=remaining_args)


def get_transforms(train):
    t = [
        transform.ResizeKPS(IMAGE_SIZE),
        transform.ToTensor(),
        transform.ImageTargetWrapper(T.Normalize(CocoSingleKPS.MEAN, CocoSingleKPS.STD)),
        transform.ConvertCocoKps(),
    ]
    if train:
        t.append(transform.RandomHorizontalFlip(0.5))
    t.append(transform.ExtractKeypoints())
    return transform.Compose(t)


coco_train = CocoSingleKPS.from_data_path(data_path, train=True, transforms=get_transforms(True))
coco_val = CocoSingleKPS.from_data_path(data_path, train=False, transforms=get_transforms(False))

num_instructions = len(coco_utils.KEYPOINTS)
model = models.resnet18(td_outplanes=64, num_instructions=num_instructions)
td_head = models.TDHead()
model = models.SequentialInstructor(model, num_instructions, td_head=td_head, skip_lateral=True)

evaluator = eval.Evaluator(original_size=IMAGE_SIZE, loss='ce')
plot = eval.Visualizer(CocoSingleKPS.MEAN, CocoSingleKPS.STD)
engine.run(model, coco_train, coco_val, evaluator, plot_fn=plot)
