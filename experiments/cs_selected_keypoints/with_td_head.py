import torchvision.transforms as T
from torch import nn

import csmodels
import engine as eng
import eval
import transform
import utils
from datasets import CocoSingleKPS

IMAGE_SIZE = 256, 256

data_path, remaining_args = utils.get_data_path()
engine = eng.Engine.command_line_init(args=remaining_args)

data_transform = transform.Compose([
    transform.ResizeKPS(IMAGE_SIZE),
    transform.extract_keypoints,
    transform.ToTensor(),
    transform.ImageTargetWrapper(T.Normalize(CocoSingleKPS.MEAN, CocoSingleKPS.STD))
])

selected_kps = ['left_eye']
coco_train = CocoSingleKPS.from_data_path(data_path, train=True, transforms=data_transform, keypoints=selected_kps)
coco_val = CocoSingleKPS.from_data_path(data_path, train=False, transforms=data_transform, keypoints=selected_kps)

num_instructions = len(selected_kps)
model = csmodels.resnet50(td_outplanes=64, num_instructions=num_instructions)
if len(selected_kps) == 1:
    model.one_iteration()


def block(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm2d(out_planes, out_planes),
        nn.ReLU()
    )


class Squeeze(nn.Module):

    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze(dim=self.dim)
        return x


td_head = nn.Sequential(
    nn.BatchNorm2d(64, 64),
    nn.ReLU(),
    nn.AvgPool2d(3, stride=2, padding=1),
    block(64, 64),
    block(64, 64),
    block(64, 32, stride=2),
    block(32, 32),
    block(32, 16, stride=2),
    block(16, 16),
    nn.Conv2d(16, 1, kernel_size=1),
    Squeeze(dim=1)
)
model = csmodels.SequentialInstructor(model, num_instructions, td_head=td_head)

train_eval = eval.Evaluator()
val_eval = eval.Evaluator()
plot = eval.Visualizer(CocoSingleKPS.MEAN, CocoSingleKPS.STD)
engine.run(model, coco_train, coco_val, train_eval, val_eval, plot_fn=plot)
