from fastai.vision import *

import parts
import utils
from models import layers
from models.upernet import get_fpn


def resize_sample(sample, size, resize_method=ResizeMethod.PAD):
    img, obj_and_part = sample
    t = (o.apply_tfms(None, size=size, resize_method=resize_method, padding_mode='zeros')
         for o in (img, obj_and_part))
    return tuple(t)


class ScaleJitterCollate:

    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, samples):
        size = random.choice(self.sizes)
        samples = [resize_sample(s, size) for s in samples]
        out = data_collate(samples)
        return out


class UperNetAdapter:
    """imitate upernet ValDataset"""

    def __init__(self):
        super().__init__()
        self.mean = [102.9801, 115.9465, 122.7717]
        self.std = [1., 1., 1.]
        self.idx = torch.LongTensor([2, 1, 0])

    def __call__(self, batch):
        img, (yb, orig) = batch

        img = img.index_select(1, self.idx.to(device=img.device))
        img = img * 255
        mean = torch.tensor(self.mean, device=img.device)
        std = torch.tensor(self.std, device=img.device)
        img = (img - mean[:, None, None]) / std[:, None, None]
        return img, (yb, orig)


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
    ckpt_dir = root.parent.resolve() / 'ckpt'
    encoder_ckpt = str(ckpt_dir / 'trained/encoder_epoch_40.pth')
    decoder_ckpt = str(ckpt_dir / 'trained/decoder_epoch_40.pth')

    fpn = get_fpn(tree, encoder_ckpt, decoder_ckpt)
    outputs = {'object': tree.n_obj, 'part': tree.n_parts}
    model = nn.Sequential(fpn, layers.SplitHead(512, outputs))
    return model


class BnFreeze(Callback):
    """Freeze moving average statistics in all non-trainable batchnorm layers."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_begin(self, **kwargs: Any) -> None:
        """Put bn layers in eval mode just after `model.train()`."""
        set_bn_eval(self.model)


def main(args):
    save = args.save

    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    model = get_model(broden_root, tree)
    adapter_tfm = UperNetAdapter()
    train_collate = ScaleJitterCollate([384, 480, 544, 608, 672])
    val_collate = ScaleJitterCollate([544])
    data = parts.get_data(broden_root, size=None, norm_stats=imagenet_stats,
                          max_rotate=None, max_zoom=1, max_warp=None, max_lighting=None,
                          bs=8, no_check=True, dl_tfms=adapter_tfm)
    data.train_dl.dl.collate_fn = train_collate
    data.valid_dl.dl.collate_fn = val_collate

    loss = parts.Loss(tree, split_func=LossSplit(tree))
    metrics = partial(parts.BrodenMetricsClbk, obj_tree=tree, restrict=True, split_func=split_func)
    learn = Learner(data, model, loss_func=loss, callback_fns=metrics, callbacks=BnFreeze(model[0]), train_bn=False)
    learn.split((learn.model[1],))
    learn.freeze()

    logger = callbacks.CSVLogger(learn, filename=args.save, )
    save_clbk = callbacks.SaveModelCallback(learn, monitor='object-P.A.', mode='max', every='epoch', name=save)
    learn.fit_one_cycle(20, 1e-2, pct_start=0.1, callbacks=[logger, save_clbk])


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    main(parser.parse_args())
