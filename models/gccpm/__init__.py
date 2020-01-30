from .load_state import load_from_mobilenet
from .model import SinglePersonPoseEstimationWithMobileNet


def gccpm(pretrain_checkpoint=None):
    model = SinglePersonPoseEstimationWithMobileNet()
    if pretrain_checkpoint:
        load_from_mobilenet(model, pretrain_checkpoint)
    return model
