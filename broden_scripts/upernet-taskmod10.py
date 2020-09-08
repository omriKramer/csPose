from fastai.vision import *

import parts
import utils
from models import cs_head, taskmod


def get_upernet(broden_root, tree, objects, sample_train=True, pos_weight=None):
    instructor = cs_head.FullHeadInstructor(tree, obj_classes=objects, sample_train=sample_train)
    model = taskmod.upernetmod(broden_root, tree, instructor)
    loss_func = cs_head.BCEObjectLoss(instructor, pos_weight=pos_weight, softmax=True)
    loss_func.to('cuda')
    return model, instructor, loss_func


def pred_func(objects, last_output, gt_size):
    obj_pred = last_output[0].clone()
    for pred, o in zip(last_output[1:], objects):
        obj_pred[:, o] = pred[:, o]

    obj_pred = utils.resize(obj_pred, gt_size)
    obj_pred = obj_pred.argmax(dim=1)
    return obj_pred, None


def val_pred(last_output, gt_size):
    obj_pred = last_output[0]
    obj_pred = utils.resize(obj_pred, gt_size)
    obj_pred = obj_pred.argmax(dim=1)
    return obj_pred, None


def main(args):

    save = args.save

    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    cls10 = ['table', 'light', 'sofa', 'bottle', 'stove', 'refrigerator', 'microwave', 'dishwasher', 'oven', 'kettle']
    obj10 = [i for i, name in enumerate(tree.obj_names) if name in cls10]

    model, instructor, loss_func = get_upernet(broden_root, tree, obj10, pos_weight=1e4)
    db = parts.upernet_data_pipeline(broden_root, bs=8)

    learn = Learner(db, model, loss_func=loss_func, callbacks=[instructor], callback_fns=[BnFreeze], train_bn=False)
    learn.split((learn.model.embeddings,))
    learn.freeze()

    logger = callbacks.CSVLogger(learn, filename=args.save)
    save_clbk = callbacks.SaveModelCallback(learn, monitor='object-P.A.', mode='max', every='epoch', name=save)
    m1 = parts.BrodenMetricsClbk(learn, obj_tree=tree, pred_func=partial(pred_func, obj10), object_only=True)
    learn.fit_one_cycle(10, (0, 1e-3), pct_start=0.1, wd=0, callbacks=[logger, save_clbk, m1])


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    main(parser.parse_args())
