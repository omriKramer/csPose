from fastai.vision import *

import models.cs_v2 as cs
import pose
from utils import DataTime

lateral_types = {
    'add': cs.conv_add_lateral,
    'mul': cs.conv_mul_lateral,
    'attention': cs.attention_lateral,
}

nets = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
}


def main(args):
    print(args)
    arch = nets[args.resnet]

    n = args.niter
    instructor = cs.RecurrentInstructor(n)
    pckh = partial(pose.Pckh, niter=n)
    loss = pose.RecurrentLoss(n)

    root = Path(__file__).resolve().parent.parent / 'LIP'
    db = pose.get_data(root, args.size, bs=args.bs)

    lateral = lateral_types[args.lateral]
    learn = cs.cs_learner(db, arch, instructor, td_c=16, pretrained=False, embedding=None, lateral=lateral,
                          loss_func=loss, callback_fns=[pckh, DataTime])
    if args.load:
        learn.load(args.load)

    logger = callbacks.CSVLogger(learn, filename=args.save)
    monitor = f'Total_{n - 1}' if n > 1 else 'Total'
    save_clbk = callbacks.SaveModelCallback(learn, monitor=monitor, mode='max', every='improvement', name=args.save)
    if args.one_cycle:
        learn.fit_one_cycle(args.epochs, args.lr, start_epoch=args.start_epoch, callbacks=[logger, save_clbk])
    else:
        epochs = args.epochs - args.start_epoch
        learn.fit(epochs, args.lr, callbacks=[logger, save_clbk])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('save', type=str)
    parser.add_argument('-n', '--niter', default=1, type=int)
    parser.add_argument('-e', '--epochs', default=60, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('-r', '--resnet', default=18, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--bs', default=64, type=int)
    parser.add_argument('-s', '--size', default=128, type=int)
    parser.add_argument('-l', '--load', default=None, type=str)
    parser.add_argument('--one-cycle', action='store_true')
    parser.add_argument('--lateral', choices=lateral_types, default='add')
    main(parser.parse_args())
