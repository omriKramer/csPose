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

    print(args)

    n = args.niter
    instructor = cs.RecurrentInstructor(n)
    pckh = partial(pose.Pckh, niter=n)
    loss = RecurrentLoss(n)

    root = Path(__file__).resolve().parent.parent / 'LIP'
    db = pose.get_data(root, args.size, bs=args.bs)

    learn = cs.cs_learner(db, model, instructor, td_c=16, pretrained=False, embedding=None,
                          loss_func=loss, callback_fns=[pckh, DataTime])
    if args.load:
        learn.load(args.load)

    logger = callbacks.CSVLogger(learn, filename=args.save)
    monitor = f'Total_{n - 1}' if n > 1 else 'Total'
    save_clbk = callbacks.SaveModelCallback(learn, monitor=monitor, mode='max', every='improvement', name=args.save)
    if args.one_cycle:
        learn.fit_one_cycle(args.epochs, args.lr, start_epoch=args.start_epoch, callbacks=[logger, save_clbk])
    else:
        learn.fit(args.epochs, args.lr, start_epoch=args.start_epoch, callbacks=[logger, save_clbk])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('save', type=str)
    parser.add_argument('-n', '--niter', default=1, type=int)
    parser.add_argument('-e', '--epochs', default=60, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('-r', '--resnet', default=18, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--bs', default=32, type=int)
    parser.add_argument('-s', '--size', default=128, type=int)
    parser.add_argument('-l', '--load', default=None, type=str)
    parser.add_argument('--one-cycle', action='store_true')
    main(parser.parse_args())
