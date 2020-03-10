# Pose Results
| Date   | Notebook    | Size    | Epochs  | Loss | Transform | PCKH  | Arch  | Pretrained | Comments |
| ------:|:-----------:| -------:| ------: | ---: | --------: | ----: |-----: | ---------: | -------: |
|        | lip_pose_cnn_learner | 128 | 10 | MSE - Regression | lr_flip | 47.6 | resnet18 | True | simple baseline, Could train longer |
| 26.01 | cs_baseline | 128->224| 55 | CE |  lr_flip | 76.3 | csResnet18 | True | simple TDBlock per BU layer, Single Instruction |
| 28.01 | cs_sample_instructions | 128->224 | 75 | CE | lr_flip | 73.3 | csResnet18 | True | simple TDBlock per BU layer. Each iteration sample one instruction\keypoint. Val loss was lower than train (not enough epochs?)
| 29.01 | bu_multilabel_loss | 128->224 | 50 | CE | lr_flip | 77.2 | csResnet18 | True | simple TDBlock per BU layer. BU detects which keypoints are present and then sends it as the instruction |
| 29.01 | bu_multilabel_noinst (a)| 128->224 | 50 | CE | lr_flip | 77.5 | csResnet18 | True | simple TDBlock per BU layer. No Instruction given. |
| 29.01 | bu_multilabel_noinst (b)| 128->224 | 50 | CE | lr_flip | 67.2 | csResnet18 | True | simple TDBlock per BU layer. No Instruction given. Detached inner state of laterals.  |
| 02.02 | unet (a) | 128->256 | 40 | CE | lr_flip | 73 | resnet-18 unet | True | fastai unet |
| 02.02 | unet (b) | 128->256 | 40 | CE | lr_flip | 76 | resnet-18 unet | True | replaced head of unet (output is two times smaller than input |
| 03.02 | unet (c) | 128->256 | 40 | CE | lr_flip | 75.5 | csunet-18 | True | replaced head of unet and BU loss|

## V2
Added augmentations and deeper TD.

| Date   | Notebook/Name    | Size    | Epochs  | Loss | PCKH  | Arch  | Pretrained | Comments |
| ------:|:-----------:| -------:| ------: | ---: | ----: |-----: | ---------: | -------: |
|        | baselineV2 | 128 | 40 | CE | 77.2 | CSResnet18 | X | |
|  | baselineV2 (b) | 128 | 30 | CE | 75.4 | csResnet18 | V | |
| 12.02 | baselineV2 (C) | 128 | 40 | CE | 77.3 | csResnet18 | X | No instructions|
| 11.02 | multilabelV2 | 128 | 40 | CE | 76.4 | csResnet18 | X | With BU Loss |
| 11.02 | multilabelV2 (b) | 128 | 40 | CE | 77.2 | csResnet18 | X | With BU Loss, No instruction |
| 15.02 | recurrent (a) | 128 | 40 | CE | 78.4 | csResnet18 | X | repeat 2 times |
| 16.02 | recurrent (b) | 128 | 40 | CE | 78.6 | csResnet18 | X | repeat 3 times |
| 16.02 | repeat4 | 128 | 40 | CE | 78.8 | csResnet18 | X | repeat 4 times |
| 16.02 | repeat8 | 128 | 40 | CE | 78.3 | csResnet18 | X | repeat 8 times |
| 16.02 | repeat-detach2 (a) | 128 | 40 | CE | 78.3 | csResnet18 | X | repeat 2 times and detach td laterals|
| 16.02 | recurrent (c) | 128 | 40 | CE | 78.6 | csResnet18 | X | repeat 3 times and detach td laterals|
| 16.02 | repeat-detach2 (b) | 128 | 40 | CE | 78.7 | csResnet18 | X | repeat 4 times and detach td laterals|
| 08.03 | baselinev2-128-40 | 128 | 40 | CE | 78.3 | csResnet50 | X | baseline with resnet50 |
| 08.03 | baselinev2-128-40-1e-3 | 128 | 40 | CE | 77.4 | csResnet50 | X | lr 1e-3 |
| 08.03 | baselinev2-128-40-5e-3 | 128 | 40 | CE | 77.6| csResnet50 | X | lr 5e-3 |
| 08.03 | baselinev2-256-100 | 256 | 100 (72) | CE | 81.0 | csResnet50 | X |lr 1e-3 |
| 08.03 | baselinev2-256-100-1e-2| 256 | 100 | CE | 75.2 | csResnet50 | X | lr 1e-2 |
| 08.03 | baselinev2-128-256-1e-3| 256 | 40 | CE | 79.8 | csResnet50 | X | loaded baselinev2-128-40 |
