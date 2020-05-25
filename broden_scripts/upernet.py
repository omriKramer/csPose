from fastai.vision import *

import parts
import utils
from models.upernet import get_upernet


def resize_sample(sample, size):
    img, obj_and_part = sample
    t = (o.apply_tfms(None, size=size, resize_method=ResizeMethod.PAD, padding_mode='zeros')
         for o in (img, obj_and_part))
    return tuple(t)


class ScaleJitterCollate:

    def __init__(self, model, train_sizes=None, eval_size=480):
        if not train_sizes:
            train_sizes = [320, 384, 480, 544, 608]

        self.train_sizes = train_sizes
        self.eval_size = eval_size
        self.model = model

    def __call__(self, samples):
        if self.model.training:
            size = random.choice(self.train_sizes)
        else:
            size = self.eval_size

        samples = [resize_sample(s, size) for s in samples]
        batched = data_collate(samples)
        return batched


class UperNetAdapter(Callback):
    """imitate upernet ValDataset"""

    def __init__(self):
        super().__init__()
        self.mean = [102.9801, 115.9465, 122.7717]
        self.std = [1., 1., 1.]
        self.idx = torch.LongTensor([2, 1, 0])

    def __call__(self, batch):
        img, yb = batch

        img = img.index_select(1, self.idx.to(device=img.device))
        img = img * 255
        mean = torch.tensor(self.mean, device=img.device)
        std = torch.tensor(self.std, device=img.device)
        img = (img - mean[:, None, None]) / std[:, None, None]
        return img, yb


class LossAdapter:

    def __init__(self, tree, loss):
        self.tree = tree
        self.loss = loss

    def __call__(self, pred, obj_gt, part_gt):
        obj_pred = pred['object']
        obj_pred = F.log_softmax(obj_pred, dim=1)

        part_pred_list, head = [], 0
        for idx_part, (object_label, o_parts) in enumerate(self.tree.obj_and_parts()):
            n_part = len(o_parts)
            x = pred['part'][:, head: head + n_part]
            x = F.log_softmax(x, dim=1)
            part_pred_list.append(x)
            head += n_part
        pred_dict = {'object': obj_pred, 'part': part_pred_list}

        obj_gt = obj_gt.squeeze(dim=1)
        part_gt = part_gt.squeeze(dim=1)

        size = obj_pred.shape[-2:]
        obj_gt = parts.resize(obj_gt, size)
        part_gt = parts.resize(part_gt, size)

        part_gt = self.tree.split_parts_gt(obj_gt, part_gt).transpose(0, 1)
        is_part = part_gt > 0
        valid = is_part.flatten(start_dim=2).any(dim=2)

        part_gt = part_gt.clone()
        part_gt[part_gt == -1] = 0
        part_gt = part_gt.sum(dim=1)

        results = self.loss(pred_dict, obj_gt, part_gt, valid)
        return results['loss']['total']


def split_func(last_output):
    obj_pred = last_output['object']
    part_pred = last_output['part']
    return obj_pred, part_pred


def main(args):
    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    model = get_upernet(tree)
    adapter_tfm = UperNetAdapter()
    scale_jitter = ScaleJitterCollate(model)
    db = parts.get_data(broden_root, size=None, norm_stats=None,
                        max_rotate=None, max_zoom=1, max_warp=None, max_lighting=None,
                        bs=args.bs, collate_fn=scale_jitter, dl_tfms=adapter_tfm)
    loss = LossAdapter(tree, model.loss_func)
    metrics = partial(parts.BrodenMetrics, obj_tree=tree, restrict=True, split_func=split_func)
    sgd = partial(optim.SGD, momentum=0.9)
    learn = Learner(db, model, loss_func=loss, callback_fns=metrics,
                    opt_func=sgd, wd=1e-4, true_wd=False, bn_wd=False)
    n = len(learn.data.train_dl)
    phase = callbacks.TrainingPhase(n * args.epochs).schedule_hp('lr', 2e-2, anneal=annealing_poly(0.9))
    sched = callbacks.GeneralScheduler(learn, [phase], start_epoch=args.start_epoch)
    logger = callbacks.CSVLogger(learn, filename=args.save, append=args.start_epoch > 0)
    save_clbk = callbacks.SaveModelCallback(learn, monitor='object-P.A.', mode='max', every='epoch', name=args.save)
    learn.fit(args.epochs, callbacks=[sched, logger, save_clbk])


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    main(parser.parse_args())
