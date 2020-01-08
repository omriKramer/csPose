import coco_utils
import engine as eng
import eval
import models
import utils
from datasets import CocoSingleKPS
from transform import get_single_kps_transforms

IMAGE_SIZE = 128, 128

args, remaining_args = utils.get_args()
engine = eng.Engine.command_line_init(args=remaining_args)
mean = CocoSingleKPS.MEAN
std = CocoSingleKPS.STD
coco_train = CocoSingleKPS.from_data_path(args.data_path, train=True,
                                          transforms=get_single_kps_transforms(True, IMAGE_SIZE, mean, std))
coco_val = CocoSingleKPS.from_data_path(args.data_path, train=False,
                                        transforms=get_single_kps_transforms(False, IMAGE_SIZE, mean, std))

num_instructions = len(coco_utils.KEYPOINTS)
model = models.resnet18(td_outplanes=64, num_instructions=num_instructions)
td_head = models.TDHead(num_channels=num_instructions)
model = models.SequentialInstructor(model, num_instructions, td_head=td_head, skip_lateral=args.skip_lateral)

evaluator = eval.Evaluator(original_size=IMAGE_SIZE, loss=args.loss)
plot = eval.Visualizer(CocoSingleKPS.MEAN, CocoSingleKPS.STD)
engine.run(model, coco_train, coco_val, evaluator, plot_fn=plot)
