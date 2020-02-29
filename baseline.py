from fastai.callbacks import SaveModelCallback
from fastai.vision import *

import models.cs_v2 as cs
import pose


class RecurrentLoss:

    def __init__(self, repeats):
        self.r = repeats

    def __call__(self, outputs, targets):
        targets = targets.repeat(1, self.r, 1)
        return pose.pose_ce_loss(outputs[1], targets)


def main(args):
    print(args)
    root = Path(__file__).resolve().parent.parent / 'LIP'
    niter = args.niter
    instructor = cs.RecurrentInstructor(niter)
    pckh = partial(pose.Pckh, niter=niter, mean=True)

    logger = partial(callbacks.CSVLogger, filename='baseline')
    db = pose.get_data(root, 64, bs=64)
    learn = cs.cs_learner(db, models.resnet18, instructor, td_c=16, pretrained=False, embedding=None,
                          loss_func=RecurrentLoss(niter), callback_fns=[pckh, logger])
    learn.fit_one_cycle(10, 1e-2)

    learn.data = pose.get_data(root, 128, bs=64)
    learn.fit_one_cycle(40, 5e-3)

    monitor = f'Total_{niter - 1}' if niter > 1 else 'Total'
    save_clbk = SaveModelCallback(learn, every='improvement', monitor=monitor, name=f'baseline-{args}', mode='max')
    learn.callbacks.append(save_clbk)
    learn.data = pose.get_data(root, 256, bs=32)
    learn.fit_one_cycle(60, 1e-4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=1, type=int)
    main(parser.parse_args())
