from fastai.vision import *

import parts
import utils
from models import CSHead, Instructor
from parts import upernet_data_pipeline


def get_model(root, tree, op):
    encoder_path, decoder_path = utils.upernet_ckpt(root)
    instructor = Instructor(tree)
    model = CSHead(instructor, tree, encoder_path, decoder_path, emb_op=op)
    return model, instructor


ops = {
    'mul': torch.mul,
    'add': torch.add,
}


def main(args):
    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    model, instructor = get_model(broden_root, tree, ops[args.op])
    db = upernet_data_pipeline(broden_root)

    metrics = partial(parts.BinaryBrodenMetrics, obj_tree=tree, thresh=0.75)
    clbks = [instructor]
    if not args.train_bn:
        clbks.append(utils.BnFreeze(model.fpn))
    learn = Learner(db, model, loss_func=instructor.loss, callbacks=clbks, callback_fns=metrics, train_bn=args.train_bn)
    learn.split((learn.model.td,))
    learn.freeze()

    utils.fit_and_log(learn, 'object-P.A', save=args.save, epochs=20, lr=1e-2, pct_start=args.pct_start)


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    parser.add_argument('--train_bn', action='store_true')
    parser.add_argument('--op', choices=('mul', 'add'), default='mul')
    main(parser.parse_args())
