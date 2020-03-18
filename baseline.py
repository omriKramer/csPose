from fastai.vision import *

import models.cs_v2 as cs
import pose
import utils
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

    monitor = f'Total_{n - 1}' if n > 1 else 'Total'
    utils.fit_and_log(learn, args, monitor)


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('-n', '--niter', default=1, type=int)
    parser.add_argument('--lateral', choices=lateral_types, default='add')
    main(parser.parse_args())
