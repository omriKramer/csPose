import argparse
import random
from time import time
from typing import Any

import fastai
import fastprogress
import numpy as np
import torch
import torch.distributed as dist
from fastai import callbacks
from fastai.core import master_bar, progress_bar
from fastai.torch_core import set_bn_eval
from fastai.vision import Callback, LearnerCallback, add_metrics, ResizeMethod, data_collate
from fastprogress.fastprogress import force_console_behavior
from torch.nn import functional as F
from torch.utils.data import DataLoader


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def my_print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = my_print


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='~/weizmann/coco/dev', help='dataset location')
    parser.add_argument('--loss', default='ce', choices=['ce', 'kl'], help='which lose to use')
    parser.add_argument('--skip-lateral', action='store_true', help='whether to skip latter connections')
    args, remaining_args = parser.parse_known_args()
    return args, remaining_args


def dataset_mean_and_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0.
    for i, (images, _) in enumerate(loader):
        batch_samples = images.shape[0]
        data = images.reshape(batch_samples, images.shape[1], -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


def one_hot2d(x, h, w):
    out = x[..., 0] * w + x[..., 1]
    out = F.one_hot(out, h * w)
    out = out.reshape(*x.shape[:-1], h, w)
    return out


class ProgressBarCtx:
    """Context manager to disable the progress update bar."""

    def __init__(self, show=True):
        self.show = show

    def __enter__(self):
        if self.show:
            return
        # silence progress bar
        fastprogress.fastprogress.NO_BAR = True
        fastai.basic_train.master_bar, fastai.basic_train.progress_bar = force_console_behavior()

    def __exit__(self, *args):
        fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar


class DataTime(LearnerCallback):
    _order = -20

    def __init__(self, learn):
        super().__init__(learn)
        self.total_time = 0.
        self.start = None

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['data_time'])

    def on_epoch_begin(self, **kwargs):
        self.total_time = 0.
        self.start = None

    def on_batch_begin(self, train, **kwargs):
        if self.start is None:
            return

        self.total_time += time() - self.start

    def on_batch_end(self, train, **kwargs):
        self.start = time()

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self.total_time / 60)


class AddTargetClbk(Callback):

    def __init__(self):
        super().__init__()

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if train:
            return {'last_input': (last_input, last_target)}


def fit_and_log(learn, monitor, save='bestmodel', epochs=40, start_epoch=0, lr=1e-2, wd=None, load=None,
                no_one_cycle=False, pct_start=0.3):
    if load:
        learn.load(load)

    logger = callbacks.CSVLogger(learn, filename=save, append=start_epoch > 0)
    save_clbk = callbacks.SaveModelCallback(learn, monitor=monitor, mode='max', every='epoch', name=save)

    if no_one_cycle:
        epochs = epochs - start_epoch
        learn.fit(epochs, lr, wd=wd, callbacks=[logger, save_clbk])
    else:
        learn.fit_one_cycle(epochs, lr, wd=wd, start_epoch=start_epoch, pct_start=pct_start,
                            callbacks=[logger, save_clbk])


def basic_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('save', type=str)
    parser.add_argument('-e', '--epochs', default=60, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('-r', '--resnet', default=34, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=None, type=float)
    parser.add_argument('--bs', default=64, type=int)
    parser.add_argument('-s', '--size', default=128, type=int)
    parser.add_argument('-l', '--load', default=None, type=str)
    parser.add_argument('--no-one-cycle', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pct-start', default=0.3, type=float)
    return parser


def basic_broden_parser():
    parser = basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    return parser


class UperNetAdapter:
    """imitate upernet ValDataset"""
    mean = [102.9801, 115.9465, 122.7717]
    std = [1., 1., 1.]

    def __init__(self):
        super().__init__()
        self.idx = torch.LongTensor([2, 1, 0])

    def __call__(self, batch):
        img, (yb, orig) = batch

        img = img.index_select(1, self.idx.to(device=img.device))
        img = img * 255
        mean = torch.tensor(self.mean, device=img.device)
        std = torch.tensor(self.std, device=img.device)
        img = (img - mean[:, None, None]) / std[:, None, None]
        return img, (yb, orig)


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


class BnFreeze(Callback):
    """Freeze moving average statistics in all non-trainable batchnorm layers."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_begin(self, **kwargs: Any) -> None:
        """Put bn layers in eval mode just after `model.train()`."""
        set_bn_eval(self.model)


class BalancingSampler:

    def __init__(self, n):
        self.count = np.ones(n)

    def reset(self):
        self.count = np.ones_like(self.count)

    def sample(self, x):
        weights = 1 / self.count[x]
        c = int(np.random.choice(x, p=weights / weights.sum()))
        self.count[c] += 1
        return c


class LearnerMetrics(LearnerCallback):
    _order = -20

    def __init__(self, learn, metrics_names):
        super().__init__(learn)
        self.metric_names = metrics_names

    def on_train_begin(self, **kwargs):
        try:
            self.learn.recorder.add_metric_names(self.metric_names)
        except AttributeError:
            print('Warning: recorder is not initialized for learner')

    def on_epoch_begin(self, **kwargs):
        self._reset()

    def _reset(self):
        raise NotImplementedError


def upernet_ckpt(root):
    ckpt_dir = root.parent.resolve() / 'ckpt'
    encoder_ckpt = str(ckpt_dir / 'trained/encoder_epoch_40.pth')
    decoder_ckpt = str(ckpt_dir / 'trained/decoder_epoch_40.pth')
    return encoder_ckpt, decoder_ckpt


def resize(x, size):
    if x.shape[-2:] == size:
        return x
    three_dim = False
    if x.ndim == 3:
        x = x[None]
        three_dim = True

    if x.dtype == torch.long:
        x = x.float()
        out = F.interpolate(x, size, mode='nearest').long()
    else:
        out = F.interpolate(x, size, mode='bilinear', align_corners=False)

    if three_dim:
        out = out.squeeze(dim=0)

    return out
