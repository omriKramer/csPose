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
    def __init__(self, std=1, ks=7, sigmoid=False):
        super().__init__()
        self.g = GaussianSmoothing(std, kernel_size=ks, scale=False, thresh=0.01)
        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid() if sigmoid else None

    def create_target_heatmaps(self, targets, size):
        targets = pose.scale_targets(targets, size).round().long()
        targets = utils.one_hot2d(targets, *size).float()
        targets = self.g(targets[:, None])[:, 0]
        return targets

    def forward(self, outputs, targets):
        v = targets[..., 2].bool()
        outputs = outputs[1][v]
        if self.sigmoid:
            outputs = self.sigmoid(outputs)

        targets = targets[..., :2][v]
        targets = self.create_target_heatmaps(targets, outputs.shape[-2:])

        return self.mse(outputs, targets)


def main(args):
    print(args)
    name = 'l2-' + str(args)
    logger = partial(callbacks.CSVLogger, filename=name)

    root = Path(__file__).resolve().parent.parent / 'LIP'
    db = pose.get_data(root, args.size, bs=32)

    loss = to_device(L2Loss(args.std, args.ks, args.sigmoid), db.device)
    instructor = RecurrentInstructor(1)
    learn = cs.cs_learner(db, models.resnet18, instructor, td_c=16, pretrained=False, embedding=None,
                          loss_func=loss, callback_fns=[pose.Pckh, logger])
    learn.fit_one_cycle(40, args.lr)
    learn.save(name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--std', default=1, type=float)
    parser.add_argument('--ks', default=7, type=int)
    parser.add_argument('--sigmoid', action='store_true')
    parser.add_argument('--lr', default=.01, type=float)
    parser.add_argument('--size', default=128, type=int)
    main(parser.parse_args())
