from fastai.vision import *

import parts
from models import taskmod
from models import cs_head
import utils


def get_upernet(broden_root, tree, objects, sample_train=True):
    instructor = cs_head.FullHeadInstructor(tree, obj_classes=objects, sample_train=sample_train)
    model = taskmod.upernetmod(broden_root, tree, instructor)
    model.embeddings.apply(taskmod.init_ones)
    return model, instructor


def main(args):
    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')

    cname = 'bench'
    obj = tree.obj_names.index(cname)
    db = parts.upernet_data_pipeline(broden_root, bs=8)

    model, instructor = get_upernet(broden_root, tree, [obj])
    loss_func = taskmod.SingleObjLoss(instructor, obj, pos_weight=args.pos_weight).to('cuda')
    callback_fns = [BnFreeze]
    learn = Learner(db, model, loss_func=loss_func, callbacks=[instructor], callback_fns=callback_fns,
                    train_bn=False)
    learn.split((learn.model.embeddings,))
    learn.freeze()

    pred_func = taskmod.TaskModPred(obj)
    metrics = taskmod.TaskmodMetrics(learn, tree.n_obj, utils.cm_pred_func(pred_func), obj)

    save = callbacks.SaveModelCallback(learn, monitor='mIoU', mode='max', name=args.save)
    logger = callbacks.CSVLogger(learn, filename=args.save)
    learn.fit_one_cycle(args.epochs, (0, 3e-3), wd=0, callbacks=[metrics, save, logger])


if __name__ == '__main__':
    parser = utils.basic_broden_parser()
    parser.add_argument('--pos-weight', type=float, default=890.)
    main(parser.parse_args())
