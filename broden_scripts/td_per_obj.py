from fastai.vision import *
import parts
import utils


def main(args):
    broden_root = args.root
    db = parts.get_data(broden_root, bs=args.bs)
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    learn = parts.part_learner(db, models.resnet34, tree, pretrained=True)
    utils.fit_and_log(learn, 'object-P.A.', **vars(args))


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    main(parser.args)
