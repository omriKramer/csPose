from fastai.vision import *

import parts
import utils
from models.taskmod import taskmod


def main(args):
    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    db10 = parts.upernet_data_pipeline(broden_root, csv_file='broden10.csv')
    cls10 = ['building', 'person', 'table', 'bicycle', 'chair', 'car', 'door', 'dog', 'cat', 'bird']
    obj10 = [i for i, name in enumerate(tree.obj_names) if name in cls10]
    model, instructor = taskmod(broden_root, tree, obj_classes=obj10)
    metrics = partial(parts.BinaryBrodenMetrics, obj_tree=tree, thresh=0.5, obj_classes=obj10)

    learn = Learner(db10, model, loss_func=instructor.loss, callbacks=[instructor], callback_fns=metrics)
    learn.split((learn.model.embeddings,))
    learn.freeze()
    utils.fit_and_log(learn, 'object-P.A', save=args.save, epochs=20, lr=1e-3, pct_start=args.pct_start)


if __name__ == '__main__':
    parser = utils.basic_broden_parser()
    main(parser.parse_args())
