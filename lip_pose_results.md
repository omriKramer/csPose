# Pose Results
| Date   | Notebook    | Size    | Epochs  | Loss | Transform | PCKH  | Arch  | Pretrained | Comments |
| ------:|:-----------:| -------:| ------: | ---: | --------: | ----: |-----: | ---------: | -------: |
|        | lip_pose_cnn_learner | 128 | 10 | MSE - Regression | lr_flip | 47.6 | resnet18 | True | simple baseline, Could train longer |
| 26.01 | cs_baseline | 128->224| 55 | CE |  lr_flip | 76.3 | csResnet18 | True | simple TDBlock per BU layer, Single Instruction |
| 28.01 | cs_sample_instructions | 128->224 | 75 | CE | lr_flip | 73.3 | csResnet18 | True | simple TDBlock per BU layer. Each iteration sample one instruction\keypoint. Val loss was lower than train (not enough epochs?)