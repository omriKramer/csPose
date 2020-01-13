import engine as eng
import eval
import models
import utils
from datasets import CocoSingleKPS
from transform import get_single_kps_transforms

IMAGE_SIZE = 128, 128

args, remaining_args = utils.get_args()
data_path = args.data_path
engine = eng.Engine.command_line_init(args=remaining_args)

selected_kps = ['left_eye', 'right_eye']

mean = CocoSingleKPS.MEAN
std = CocoSingleKPS.STD
coco_train = CocoSingleKPS.from_data_path(
    data_path,
    train=True,
    transforms=get_single_kps_transforms(True, IMAGE_SIZE, mean, std, keypoints=selected_kps)
)
coco_val = CocoSingleKPS.from_data_path(
    data_path,
    train=False,
    transforms=get_single_kps_transforms(False, IMAGE_SIZE, mean, std, keypoints=selected_kps)
)

num_instructions = len(selected_kps)
model = models.resnet18(td_outplanes=64, num_instructions=num_instructions)

td_head = models.TDHead(num_channels=num_instructions)
model = models.SequentialInstructor(model, num_instructions, td_head=td_head, skip_lateral=args.skip_lateral)

evaluator = eval.Evaluator(original_size=IMAGE_SIZE, loss=args.loss)
plot = eval.Visualizer(mean, std)
engine.run(model, coco_train, coco_val, evaluator, plot_fn=plot)
