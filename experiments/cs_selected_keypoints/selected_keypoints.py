import torchvision.transforms as T

import engine as eng
import eval
import models
import transform
import utils
from datasets import CocoSingleKPS

IMAGE_SIZE = 256, 256

data_path, remaining_args = utils.get_args()
engine = eng.Engine.command_line_init(args=remaining_args)

data_transform = transform.Compose([
    transform.ResizeKPS(IMAGE_SIZE),
    transform.extract_keypoints,
    transform.ToTensor(),
    transform.ImageTargetWrapper(T.Normalize(CocoSingleKPS.MEAN, CocoSingleKPS.STD))
])

selected_kps = ['left_eye', 'right_eye']
coco_train = CocoSingleKPS.from_data_path(data_path, train=True, transforms=data_transform, keypoints=selected_kps)
coco_val = CocoSingleKPS.from_data_path(data_path, train=False, transforms=data_transform, keypoints=selected_kps)

num_instructions = len(selected_kps)
model = models.resnet50(td_outplanes=1, num_instructions=num_instructions)
if len(selected_kps) == 1:
    model.one_iteration()
model = models.SequentialInstructor(model, num_instructions)

train_eval = eval.Evaluator()
val_eval = eval.Evaluator()
plot = eval.Visualizer(CocoSingleKPS.MEAN, CocoSingleKPS.STD)
engine.run(model, coco_train, coco_val, train_eval, val_eval, plot_fn=plot)
