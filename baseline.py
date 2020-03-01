from fastai.callbacks import SaveModelCallback
from fastai.vision import *

import models.cs_v2 as cs
import pose
from utils import DataTime


class RecurrentLoss:

    def __init__(self, repeats):
        self.r = repeats

    def __call__(self, outputs, targets):
        targets = targets.repeat(1, self.r, 1)
        return pose.pose_ce_loss(outputs[1], targets)


def main(n=1, e64=10, e128=40, e256=60, resnet=18):
    if resnet == 18:
        model = models.resnet18
    elif resnet == 50:
        model = models.resnet50
    else:
        raise ValueError
    name = f'n={n}_e64={e64}_e128={e128}_e256={e256}'
    print(name)
    root = Path(__file__).resolve().parent.parent / 'LIP'
    instructor = cs.RecurrentInstructor(n)
    pckh = partial(pose.Pckh, niter=n, mean=True)

    db = pose.get_data(root, 64, bs=64)
    logger = partial(callbacks.CSVLogger, filename=f'baseline-{name}')
    monitor = f'Total_{n - 1}' if n > 1 else 'Total'
    save_clbk = partial(SaveModelCallback, every='improvement', monitor=monitor, name=f'baseline-{name}', mode='max')
    learn = cs.cs_learner(db, model, instructor, td_c=16, pretrained=False, embedding=None,
                          loss_func=RecurrentLoss(n), callback_fns=[pckh, logger, save_clbk, DataTime])
    if e64 > 0:
        learn.fit_one_cycle(e64, 1e-2)

    if e128 > 0:
        learn.data = pose.get_data(root, 128, bs=64)
        learn.fit_one_cycle(e128, 5e-3)

    learn.data = pose.get_data(root, 256, bs=32)
    learn.fit_one_cycle(e256, 1e-4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=1, type=int)
    parser.add_argument('--e64', default=10, type=int)
    parser.add_argument('--e128', default=40, type=int)
    parser.add_argument('--e256', default=60, type=int)
    parser.add_argument('-r', '--resnt', default=18, type=int)
    args = parser.parse_args()
    main(**vars(args))
