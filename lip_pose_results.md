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