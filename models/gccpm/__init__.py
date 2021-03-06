from .load_state import load
from .model import SinglePersonPoseEstimationWithMobileNet


def gccpm(pretrain_checkpoint=None):
    net = SinglePersonPoseEstimationWithMobileNet()
    if pretrain_checkpoint:
        load(net, pretrain_checkpoint)
    return net
