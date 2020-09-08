from fastai.vision import *

import parts
import utils
from models.taskmod import taskmod


def init_ones(m):
    if type(m) == nn.Embedding:
        nn.init.ones_(m.weight)


confid = {
    easy
}
obj_sets = {
    'easy': ['building', 'person', 'table', 'chair', 'car', 'door', 'dog', 'cat', 'bicycle', 'bird'],
    'hard': ['table', 'light', 'sofa', 'bottle', 'stove', 'refrigerator', 'microwave', 'dishwasher', 'oven', 'kettle'],
}


def main(args):
    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    db10 = parts.upernet_data_pipeline(broden_root, csv_file='broden10.csv')
    cls10 = obj_sets[args.obj_set]
    obj10 = [tree.obj_names.index(name) for name in cls10]
    model, instructor = taskmod(broden_root, tree, obj_classes=obj10, full_head=args.full_head)
    if args.fill_ones:
        model.embeddings.apply(init_ones)

    metrics = partial(parts.BinaryBrodenMetrics, obj_tree=tree, thresh=0.5, obj_classes=obj10)
    learn = Learner(db10, model, loss_func=instructor.loss, callbacks=[instructor], callback_fns=metrics)
    learn.split((learn.model.embeddings,))
    learn.freeze()
    utils.fit_and_log(learn, 'object-P.A', save=args.save, epochs=20, lr=args.lr, pct_start=args.pct_start,
                      start_epoch=args.start_epoch)


if __name__ == '__main__':
    parser = utils.basic_broden_parser()
    parser.add_argument('--full-head', action='store_true')
    parser.add_argument('--fill-ones', action='store_true')
    parser.add_argument('--obj-set', choices=('easy', 'hard'), default='easy')
    main(parser.parse_args())
