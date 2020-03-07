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


def main(args):
    if args.resnet == 18:
        model = models.resnet18
    elif args.resnet == 50:
        model = models.resnet50
    else:
        raise ValueError

    name = 'baseline-' + str(args)
    print(name)
    n = args.niter
    root = Path(__file__).resolve().parent.parent / 'LIP'
    instructor = cs.RecurrentInstructor(n)
    pckh = partial(pose.Pckh, niter=n)

    db = pose.get_data(root, 256, bs=64)

    learn = cs.cs_learner(db, model, instructor, td_c=16, pretrained=False, embedding=None,
                          loss_func=RecurrentLoss(n), callback_fns=[pckh, DataTime])

    logger = callbacks.CSVLogger(learn, filename=name, append=args.start_epoch > 0)
    monitor = f'Total_{n - 1}' if n > 1 else 'Total'
    save_clbk = callbacks.SaveModelCallback(learn, monitor=monitor, mode='max', every='epoch', name=name)
    learn.fit_one_cycle(args.epochs, args.lr, start_epoch=args.start_epoch, callbacks=[logger, save_clbk])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--niter', default=1, type=int)
    parser.add_argument('-e', '--epochs', default=60, type=int)
    parser.add_argument('-s', '--start-epoch', default=0, type=int)
    parser.add_argument('-r', '--resnet', default=18, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--bs', default=32, type=int)
    main(parser.parse_args())
