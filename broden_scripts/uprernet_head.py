from fastai.vision import *

import parts
import utils
from models import layers
from models.upernet import get_fpn
from utils import UperNetAdapter, ScaleJitterCollate, BnFreeze


class LossSplit:

    def __init__(self, tree):
        self.tree = tree

    def __call__(self, output):
        obj_pred, part_pred = output['object'], output['part']
        part_pred_dict = {}
        for o, (start, end) in self.tree.obj2part_idx.items():
            part_pred_dict[o] = part_pred[:, start:end]
        return obj_pred, part_pred_dict


def split_func(last_output):
    obj_pred = last_output['object']
    part_pred = last_output['part']
    return obj_pred, part_pred


def get_model(root, tree):
    encoder_ckpt, decoder_ckpt = utils.upernet_ckpt(root)
    fpn = get_fpn(tree, encoder_ckpt, decoder_ckpt)
    outputs = {'object': tree.n_obj, 'part': tree.n_parts}
    model = nn.Sequential(fpn, layers.SplitHead(512, outputs))
    return model


def main(args):
    save = args.save

    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    model = get_model(broden_root, tree)
    db = parts.upernet_data_pipeline(broden_root)

    loss = parts.Loss(tree, split_func=LossSplit(tree))
    metrics = partial(parts.BrodenMetricsClbk, obj_tree=tree, restrict=True, split_func=split_func)
    learn = Learner(db, model, loss_func=loss, callback_fns=metrics, train_bn=args.train_bn)
    if not args.train_bn:
        learn.callbacks.append(BnFreeze(model[0]))

    learn.split((learn.model[1],))
    learn.freeze()

    logger = callbacks.CSVLogger(learn, filename=args.save)
    save_clbk = callbacks.SaveModelCallback(learn, monitor='object-P.A.', mode='max', every='epoch', name=save)
    learn.fit_one_cycle(20, 1e-2, pct_start=0.1, callbacks=[logger, save_clbk])


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    parser.add_argument('--train_bn', action='store_true')
    main(parser.parse_args())
