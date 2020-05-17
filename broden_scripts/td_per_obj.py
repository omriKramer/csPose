from fastai.vision import *
import parts
import utils


def main(args):
    broden_root = Path(args.root).resolve()
    db = parts.get_data(broden_root, bs=args.bs)
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    learn = parts.part_learner(db, models.resnet34, tree, pretrained=args.pretrained, sample_one=args.sample_one)
    lr = args.lr
    if args.lr0 is not None:
        lr = slice(args.lr0, lr)
    utils.fit_and_log(learn, 'object-P.A.', save=args.save, epochs=args.epochs, start_epoch=args.start_epoch,
                      lr=lr, wd=args.wd, load=args.load, no_one_cycle=args.no_one_cycle)


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    parser.add_argument('--sample_one', action='store_true')
    main(parser.parse_args())
