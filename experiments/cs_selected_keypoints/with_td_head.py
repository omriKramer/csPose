import torchvision.transforms as T
from torch import nn

import csmodels
import engine as eng
import eval
import transform
import utils
from datasets import CocoSingleKPS

IMAGE_SIZE = 128, 128

data_path, remaining_args = utils.get_data_path()
engine = eng.Engine.command_line_init(args=remaining_args)

selected_kps = ['left_eye']
train_filter_kps = ['left_eye', 'right_eye']


def get_transforms(train):
    t = [
        transform.ResizeKPS(IMAGE_SIZE),
        transform.ToTensor(),
        transform.ImageTargetWrapper(T.Normalize(CocoSingleKPS.MEAN, CocoSingleKPS.STD)),
        transform.ConvertCocoKps(),
    ]
    if train:
        t.append(transform.RandomHorizontalFlip(0.5))
    t.append(transform.ExtractKeypoints(selected_kps))
    return transform.Compose(t)


coco_train = CocoSingleKPS.from_data_path(data_path, train=True, transforms=get_transforms(True),
                                          keypoints=train_filter_kps)
coco_val = CocoSingleKPS.from_data_path(data_path, train=False, transforms=get_transforms(False),
                                        keypoints=selected_kps)

num_instructions = len(selected_kps)
model = csmodels.resnet18(td_outplanes=64, num_instructions=num_instructions)
if len(selected_kps) == 1:
    model.one_iteration()


def block(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm2d(out_planes, out_planes),
        nn.ReLU()
    )


class TDHead(nn.Module):

    def __init__(self):
        super(TDHead, self).__init__()
        self.head = nn.Sequential(
            nn.BatchNorm2d(64, 64),
            nn.ReLU(),
            block(64, 64),
            block(64, 32, stride=2),
            block(32, 32),
            block(32, 16, stride=2),
            block(16, 16),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.head(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x.squeeze(dim=1)
        return x


td_head = TDHead()
model = csmodels.SequentialInstructor(model, num_instructions, td_head=td_head, one_hot=True)

evaluator = eval.Evaluator(original_size=IMAGE_SIZE, loss='kl')
plot = eval.Visualizer(CocoSingleKPS.MEAN, CocoSingleKPS.STD)
engine.run(model, coco_train, coco_val, evaluator, plot_fn=plot)
