from fastai.vision import *

import models.cs_v2 as cs
import pose
import utils
from models.layers import GaussianSmoothing


class RecurrentInstructor(cs.BaseInstructor):
    def __init__(self, repeats):
        self.repeats = repeats
        self.i = 0

    def on_batch_begin(self, **kwargs):
        self.i = 0

    def next_inst(self, last_bu):
        self.i += 1
        state = {'continue': self.i < self.repeats}
        return None, state


class L2Loss(Module):
    def __init__(self, std=1, ks=7):
        super().__init__()
        self.g = GaussianSmoothing(std, kernel_size=ks, scale=False, thresh=0.01)
        self.mse = nn.MSELoss()

    def create_target_heatmaps(self, targets, size):
        targets = pose.scale_targets(targets, size).round().long()
        targets = utils.one_hot2d(targets, *size).float()
        targets = self.g(targets[:, None])[:, 0]
        return targets

    def forward(self, outputs, targets):
        v = targets[..., 2].bool()
        outputs = outputs[1][v]
        targets = targets[..., :2][v]

        targets = self.create_target_heatmaps(targets, outputs.shape[-2:])
        return self.mse(outputs, targets)


def main(args):
    print(args)
    root = Path(__file__).parent.parent / 'LIP'
    db = pose.get_data(root, 128, bs=32)
    c_out = 16
    loss = to_device(L2Loss(args.std, args.ks), db.device)
    instructor = RecurrentInstructor(1)
    learn = cs.cs_learner(db, models.resnet18, instructor, td_c=c_out, pretrained=False, embedding=None,
                          loss_func=loss, callback_fns=pose.Pckh)
    lr = 1e-2
    learn.fit_one_cycle(40, lr)
    name = 'l2-' + str(args)
    learn.save(name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--std', default=1)
    parser.add_argument('--ks', default=7)
    main(parser.parse_args())
