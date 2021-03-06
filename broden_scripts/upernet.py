from fastai.vision import *

import parts
import utils
from models.upernet import get_upernet


def resize_sample(sample, size):
    img, obj_and_part = sample
    t = (o.apply_tfms(None, size=size, resize_method=ResizeMethod.PAD, padding_mode='zeros')
         for o in (img, obj_and_part))
    return tuple(t)


class MyDataLoader(DeviceDataLoader):

    def __init__(self, dl, device, tfms=None, pre_proc_batch=None):
        super().__init__(dl, device, tfms, collate_fn=lambda x: x)
        self.pre_proc_batch = pre_proc_batch

    def proc_batch(self, b):
        if self.pre_proc_batch:
            b = self.pre_proc_batch(b)
        b = data_collate(b)
        b = super().proc_batch(b)
        return b

    @classmethod
    def from_device_dl(cls, ddl, pre_proc_batch):
        return cls(ddl.dl, ddl.device, tfms=ddl.tfms, pre_proc_batch=pre_proc_batch)


class ScaleJitter:

    def __init__(self, train_sizes=None, eval_size=480):
        if not train_sizes:
            train_sizes = [320, 384, 480, 544, 608]

        self.train_sizes = train_sizes
        self.eval_size = eval_size

    def tfm_train(self, samples):
        size = random.choice(self.train_sizes)
        samples = [resize_sample(s, size) for s in samples]
        return samples

    def tfm_val(self, samples):
        samples = [resize_sample(s, self.eval_size) for s in samples]
        return samples


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
        obj_gt = utils.resize(obj_gt, size)
        part_gt = utils.resize(part_gt, size)

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
    save = args.save

    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    model = get_upernet(tree)
    adapter_tfm = UperNetAdapter()
    scale_jitter = ScaleJitter()
    db = parts.get_data(broden_root, size=None, norm_stats=None,
                        max_rotate=None, max_zoom=1, max_warp=None, max_lighting=None,
                        bs=4, dl_tfms=[adapter_tfm], collate_fn=lambda x: x)
    db.train_dl = MyDataLoader.from_device_dl(db.train_dl, pre_proc_batch=scale_jitter.tfm_train)
    db.valid_dl = MyDataLoader.from_device_dl(db.valid_dl, pre_proc_batch=scale_jitter.tfm_val)

    loss = LossAdapter(tree, model.loss_func)
    metrics = partial(parts.BrodenMetricsClbk, obj_tree=tree, restrict=True, split_func=split_func)
    sgd = partial(optim.SGD, momentum=0.9)
    learn = Learner(db, model, loss_func=loss, callback_fns=metrics,
                    opt_func=sgd, wd=1e-4, true_wd=False, bn_wd=False)
    if not args.start_epoch and args.auto_continue:
        start_epoch = find_last_epoch(learn, save) + 1
    else:
        start_epoch = args.start_epoch

    n = len(learn.data.train_dl)
    phase = callbacks.TrainingPhase(n * args.epochs).schedule_hp('lr', 2e-2, anneal=annealing_poly(0.9))
    sched = callbacks.GeneralScheduler(learn, [phase], start_epoch=start_epoch)
    logger = callbacks.CSVLogger(learn, filename=args.save, append=start_epoch > 0)
    save_clbk = callbacks.SaveModelCallback(learn, monitor='object-P.A.', mode='max', every='epoch', name=save)
    learn.fit(args.epochs, callbacks=[sched, logger, save_clbk])


def find_last_epoch(learn, save):
    p = Path(learn.path).resolve() / 'models'
    last_epoch = 0
    pat = re.compile(fr'{save}_(\d+)\.pth')
    for f in p.iterdir():
        match = re.match(pat, f.name)
        if match:
            last_epoch = max(last_epoch, int(match.group(1)))
    return last_epoch


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    parser.add_argument('--auto-continue', action='store_true')
    main(parser.parse_args())
