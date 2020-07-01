from fastai.vision import *

import parts
import utils
from models.taskmod import taskmod
from parts import upernet_data_pipeline


def main(args):
    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    db = upernet_data_pipeline(broden_root)
    model, instructor = taskmod(broden_root, tree)
    metrics = partial(parts.BinaryBrodenMetrics, obj_tree=tree, thresh=0.5)
    learn = Learner(db, model, loss_func=instructor.loss, callbacks=[instructor], callback_fns=metrics)
    learn.split((learn.model.embeddings,))
    learn.freeze()

    utils.fit_and_log(learn, 'object-P.A', save=args.save, epochs=20, lr=1e-3, pct_start=args.pct_start)


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    main(parser.parse_args())
