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

| Date   | Notebook/Name | Size    | Epochs  | Loss | PCKH  | Arch  | Pretrained | lr | load | Comments |
| ------:|:-------------:| -------:| -------:| ----:| -----:| -----:| ----------:| ---| ----:| --------:|
|        | baselineV2 | 128 | 40 | CE | 77.2 | CSResnet18 | X | - | no |  |
|  | baselineV2 (b) | 128 | 30 | CE | 75.4 | csResnet18 | V | |
| 12.02 | baselineV2 (C) | 128 | 40 | CE | 77.3 | csResnet18 | X | | | No instructions|
| 11.02 | multilabelV2 | 128 | 40 | CE | 76.4 | csResnet18 | X | | |With BU Loss |
| 11.02 | multilabelV2 (b) | 128 | 40 | CE | 77.2 | csResnet18 | X | | | With BU Loss, No instruction |
| 15.02 | recurrent (a) | 128 | 40 | CE | 78.4 | csResnet18 | X | | |repeat 2 times |
| 16.02 | recurrent (b) | 128 | 40 | CE | 78.6 | csResnet18 | X | | |repeat 3 times |
| 16.02 | repeat4 | 128 | 40 | CE | 78.8 | csResnet18 | X | | |repeat 4 times |
| 16.02 | repeat8 | 128 | 40 | CE | 78.3 | csResnet18 | X | | |repeat 8 times |
| 16.02 | repeat-detach2 (a) | 128 | 40 | CE | 78.3 | csResnet18 | X | | |repeat 2 times and detach td laterals|
| 16.02 | recurrent (c) | 128 | 40 | CE | 78.6 | csResnet18 | X | | |repeat 3 times and detach td laterals|
| 16.02 | repeat-detach2 (b) | 128 | 40 | CE | 78.7 | csResnet18 | X |  | |repeat 4 times and detach td laterals|
| 08.03 | baselinev2-128-40 | 128 | 40 | CE | 78.3 | csResnet50 | X | 1e-2 | |baseline with resnet50 |
| 08.03 | baselinev2-128-40-1e-3 | 128 | 40 | CE | 77.4 | csResnet50 | X |1e-3 |
| 08.03 | baselinev2-128-40-5e-3 | 128 | 40 | CE | 77.6| csResnet50 | X | 5e-3 |
| 08.03 | baselinev2-256-100 | 256 | 100 (72) | CE | 81.0 | csResnet50 | X | 1e-3 |
| 08.03 | baselinev2-256-100-1e-2| 256 | 100 | CE | 75.2 | csResnet50 | X | 1e-2 |
| 08.03 | baselinev2-128-256-1e-3| 256 | 40 | CE | 79.8 | csResnet50 | X | 1e-3 | baselinev2-128-40 |
| 10.03 | baseline2-128-40-fit| 128 | 40 | CE | 66.4 | csResnet50 | X | | | didn't use one_cycle |
| 11.03 | baseline2-128-40-fit-1e-3| 128 | 40 | CE | 75.4 | csResnet50 | X | 1e-3 | | didn't use one_cycle |
| 10.03 | baseline2-128-40-5e-3 | 128 | 40 | CE | 78.3 | csResnet50 | X | 5e-3 | | |
| 11.03 | baseline2-128-256-1e-4 | 256 | 40 | CE | 78.2 | csResnet50 | X | 1e-4 | baseline2-128-40-5e-3  |
| 11.03 | baseline2-128-256-7e-4 | 256 | 40 | CE | 79.9 | csResnet50 | X | 7e-4 | baseline2-128-40-5e-3  |
| 10.03 | baseline2-256-60 | 256 | 60 | CE | 81.3 | csResnet50 | X | 1e-3 |
| 10.03 | baseline2-256-60-5e-3 | 256 | 60 | CE | 80.5 | csResnet50 | X | 5e-3 |
| 11.03 | baseline2-256-70-5e-3 | 256 | 70 | CE | 80.2 | csResnet50 | X | 5e-3 |
| 13.03 | baseline2-256-70-1e-3 | 256 | 70 | CE | _ | csResnet50 | X | 1e-3 |
| 11.03 | mul-lateral-128-40-1e-2 | 128 | 40 | CE | 76.6 | csResnet50 | X | 1e-2 | | conv-multiply lateral|
| 11.03 | mul-lateral-128-40-5e-3 | 128 | 40 | CE | 76.4 | csResnet50 | X | 5e-3 | | conv-multiply lateral|
| 11.03 | mul-lateral-128-60-5e-3 | 128 | 60 | CE | - | csResnet50 | X | 5e-3 | | conv-multiply lateral|
| 11.03 | mul-lateral-128-40-1e-3 | 128 | 40 | CE | 75.5 | csResnet50 | X | 1e-3 | | conv-multiply lateral|
| 11.03 | attention-lateral-128-40-1e-3 | 128 | 40 | CE | 73.6 | csResnet50 | X | 1e-3 | | attention lateral|
| 11.03 | attention-lateral-128-40-5e-3 | 128 | 40 | CE | 72.4 | csResnet50 | X | 5e-3 | | attention lateral|


## V3
Changed TDBlock.

| Date   | Notebook/Name | Size    | Epochs  | Loss | PCKH  | Arch  | Pretrained | lr | load | Comments |
| ------:|:-------------:| -------:| -------:| ----:| -----:| -----:| ----------:| ---| ----:| --------:|
| 13.03  | baseline3-128-40-1e-3 | 128 | 40 | CE | 77.4 | CSResnet50 | X | 1e-3 |  |  |
| 13.03   | baseline3-128-40-5e-3 | 128 | 40 | CE | 78.0 | CSResnet50 | X | 5e-3 | | |
| 14.03   | baseline3-128-40-1e-2 | 128 | 40 | CE | 78.4 | CSResnet50 | X | 1e-2 | | |
| 14.03   | baseline3-r34-128-40-1e-2 | 128 | 40 | CE | 78.4 | CSResnet34 | X | 1e-2 |  | |
| 14.03   | baseline3-r34-128-40-5e-3 | 128 | 40 | CE | 77.0 | CSResnet34 | X | 5e-3 |  | |
| 14.03   | baseline3-r34-128-40-1e-3 | 128 | 40 | CE | 78.2 | CSResnet34 | X | 1e-3 |  |
| 15.03   | mul-r34-128-40-1e-3 | 128 | 40 | CE | 74.8 | CSResnet34 | X | 1e-3 |  | conv-multiply lateral |
| 15.03   | mul-r34-128-40-1e-2 | 128 | 40 | CE | 75.2 | CSResnet34 | X | 1e-2 | | conv-multiply lateral |
| 15.03   | mul-r34-128-60-1e-2 | 128 | 40 | CE | 75.2 | CSResnet34 | X | 1e-2 | | conv-multiply lateral |
| 15.03   | baseline3-128-256-r34-lr-1e-3 | 256 | 40 | CE | 82.1 | CSResnet34 | X | 1e-3 | baseline3-r34-128-40-1e-2 | |
| 15.03   | baseline3-128-256-r34-1e-3-60 | 256 | 60 | CE | 81.9 | CSResnet34 | X | 1e-3 | baseline3-r34-128-40-1e-2 | |
| 15.03   | baseline3-128-256-r34-lr-5e-4 | 256 | 40 | CE | 81.8 | CSResnet34 | X | 5e-4 | baseline3-r34-128-40-1e-2 | |
| 15.03   | baseline3-256-r34-lr-1e-3 | 256 | 50 | CE | 81.4 | CSResnet34 | X | 1e-3 | | |
| 15.03   | baseline3-256-r34-lr-5e-3 | 256 | 50 | CE | 81.6 | CSResnet34 | X | 5e-3 | | |
| 15.03   | baseline3-256-r34-8e-3 | 256 | 50 | CE | 81.8 | CSResnet34 | X | 8e-3 | | |
| 16.03   | baseline3-256-r34-1e-2 | 256 | 50 | CE | 81.9 | CSResnet34 | X | 1e-2 | | |
| 17.03   | baseline3-256-r34-1e-2-60 | 256 | 60 | CE | 81.7 | CSResnet34 | X | 1e-2 | | |
| 18.03   | baseline3-r34-256-80-1e-1-wd | 256 | 80 | CE | - | CSResnet34 | X | 1e-1 | | wd 1e-4 |
| 17.03   | gradual-stage2-1e-3 | 256 | 10 | CE | 82.45 | CSResnet34 | X | 1e-3 |recurrent-gradual | |
| 17.03   | gradual-stage2-5e-4 | 256 | 10 | CE | 82.47 | CSResnet34 | X | 5e-4 |recurrent-gradual | |
| 17.03   | gradual-stage2-1e-4 | 256 | 10 | CE | 82.58 | CSResnet34 | X | 1e-4 |recurrent-gradual | |
| 18.03   | gradual-stage2-40-1e-4 | 256 | 40 | CE | - | CSResnet34 | X | 1e-4 |recurrent-gradual | |
| 18.03   | gradual-stage2-40-1e-3 | 256 | 40 | CE | - | CSResnet34 | X | 1e-3 |recurrent-gradual | |

