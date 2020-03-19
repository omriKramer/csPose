import itertools
from abc import ABC
from typing import Callable, Tuple, Any, Union, List

import fastai.vision as fv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from . import laterals


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _get_sfs_idxs(sizes: fv.Sizes) -> List[int]:
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    if feature_szs[0] != feature_szs[1]: sfs_idxs = [0] + sfs_idxs
    return sfs_idxs


def out_channels(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        return m.out_channels
    if isinstance(m, nn.Linear):
        return m.out_features
    for child in reversed(list(m.modules())):
        c = out_channels(child)
        if c:
            return c

    return None


class TDBlock(nn.Module):

    def __init__(self, upsample, mode='nearest'):
        super().__init__()
        self.upsample = upsample
        self.mode = mode

    def maybe_upsample(self, x, identity):
        if self.upsample:
            identity = self.upsample(identity)
            if x.shape[-1] != identity.shape[-1]:
                x = F.interpolate(x, scale_factor=2, mode=self.mode)

        return x, identity


class TDBasicBlock(TDBlock):

    def __init__(self, ni, nf, upsample=None):
        super().__init__(upsample)
        self.conv1 = conv3x3(ni, ni)
        self.bn1 = nn.BatchNorm2d(ni)
        self.conv2 = conv3x3(ni, nf)
        self.bn2 = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out, identity = self.maybe_upsample(out, identity)
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class TDBottleNeck(TDBlock):
    expansion = 4

    def __init__(self, ni, nf, upsample=None):
        super().__init__(upsample)
        width = ni // self.expansion
        self.conv1 = conv1x1(ni, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, nf)
        self.bn3 = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(identity)))
        out, identity = self.maybe_upsample(out, identity)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)
        return out


class TDHead(nn.Sequential):
    def __init__(self, fi, fn):
        super().__init__(fv.conv_layer(fi, fi),
                         fv.conv2d(fi, fn, ks=1))


class CounterStream(nn.Module):
    def __init__(self, bu, instructor, td_c=1, bu_c=0, detach=False, td_detach=None, td_laterals=True,
                 add_td_out=False, detach_td_out=True,
                 embedding=fv.embedding, img_size: Tuple[int, int] = (256, 256), lateral=laterals.conv_add_lateral):
        super().__init__()
        # first few layers are not in a block, we group all layers up through MaxPool to a single block
        concat_idx = 4
        self.ifn = bu[:concat_idx]
        self.bu_body = nn.Sequential(*bu[concat_idx:])
        bu = [bu[:concat_idx]] + list(itertools.chain(*bu[concat_idx:]))
        bu = nn.Sequential(*bu)

        szs = fv.learner.model_sizes(bu, img_size)
        td_szs = list(reversed(szs))
        td = []
        if isinstance(bu[1], torchvision.models.resnet.BasicBlock):
            td_block = TDBasicBlock
        elif isinstance(bu[1], torchvision.models.resnet.Bottleneck):
            td_block = TDBottleNeck
        else:
            raise ValueError
        for sz_in, sz_out in zip(td_szs, td_szs[1:]):
            upsample = None
            if sz_in[-1] != sz_out[-1]:
                upsample = nn.Sequential(
                    fv.Lambda(lambda x: F.interpolate(x, scale_factor=2, mode='nearest')),
                    fv.conv_layer(sz_in[1], sz_out[1], ks=1, use_activ=False)
                )
            elif sz_in[1] != sz_out[1]:
                upsample = fv.conv_layer(sz_in[1], sz_out[1], ks=1, use_activ=False)
            td.append(td_block(sz_in[1], sz_out[1], upsample=upsample))

        td_layered = self._group_td(td)
        self.td = nn.Sequential(*td_layered, TDHead(td_szs[-1][1], td_c))

        channels = [s[1] for s in szs]
        td.append(self.td[-1])
        self.laterals = laterals.create_laterals(lateral, bu[:-1], td[1:], channels[:-1], detach=detach)
        if td_laterals:
            td_detach = td_detach if td_detach is not None else detach
            bu_laterals = laterals.create_laterals(lateral, td[:-1], bu[1:], reversed(channels[:-1]), detach=td_detach)
            self.laterals.extend(bu_laterals)

        if add_td_out:
            hm_lat = laterals.heatmap_add_lateral(self.td[-1], self.bu_body[0], td_c, channels[0], detach=detach_td_out)
            self.laterals.append(hm_lat)

        self.emb = embedding(instructor.n_inst, channels[-1]) if embedding else None
        self.bu_head = fv.create_head(channels[-1] * 2, bu_c) if bu_c else None
        self.instructor = instructor
        self.instructor.on_init_end(self)

    def _group_td(self, td):
        """group TDBlocks to mirror the layer groups in the BU network"""
        layer_len = [len(layer) for layer in self.bu_body]
        layer_len.reverse()
        end_idx = np.cumsum(layer_len)
        start_idx = np.roll(end_idx, 1)
        start_idx[0] = 0
        td_layered = []
        for start, end in zip(start_idx, end_idx):
            td_layered.append(nn.Sequential(*td[start:end]))
        return td_layered

    def clear(self):
        for lateral in self.laterals:
            del lateral.origin_out
            lateral.origin_out = None

    def forward(self, img):
        self.instructor.on_forward_begin(self)
        img_features = self.ifn(img)

        td_out, bu_out = [], []
        while self.instructor.on_bu_body_begin(self):

            last_bu = self.bu_body(img_features)
            if self.instructor.on_bu_pred_begin(self) and self.bu_head:
                bu_out.append(self.bu_head(last_bu))

            inst = self.instructor.on_td_begin(self, img_features, last_bu, bu_out, td_out)
            if self.emb:
                last_bu = last_bu * self.emb(inst)[..., None, None]
            td_out.append(self.td(last_bu))

            self.instructor.i += 1

        bu_out = torch.cat(bu_out, dim=1) if bu_out else None
        td_out = torch.cat(td_out, dim=1)
        return self.instructor.on_forward_end(bu_out, td_out)


def cs_learner(data: fv.DataBunch, arch: Callable, instructor, td_c=1, bu_c=0, td_laterals=True, embedding=fv.embedding,
               detach=False, td_detach=None, lateral=laterals.conv_add_lateral, add_td_out=False, detach_td_out=True,
               pretrained: bool = True, cut: Union[int, Callable] = None, **learn_kwargs: Any) -> fv.Learner:
    """Build Counter Stream learner from `data` and `arch`."""
    body = fv.create_body(arch, pretrained, cut)
    size = next(iter(data.train_dl))[0].shape[-2:]
    model = fv.to_device(
        CounterStream(body, instructor, td_c=td_c, bu_c=bu_c, img_size=size, embedding=embedding,
                      td_laterals=td_laterals, detach=detach, td_detach=td_detach, lateral=lateral,
                      add_td_out=add_td_out, detach_td_out=detach_td_out),
        data.device)
    learn = fv.Learner(data, model, **learn_kwargs)
    split = len(learn.model.laterals) // 2 + 1
    learn.split((learn.model.laterals[split],))
    if pretrained:
        learn.freeze()
    return learn


class BaseInstructor(ABC):
    def __init__(self):
        self.i = 0

    def on_forward_begin(self, model):
        self.i = 0
        model.clear()

    def on_bu_body_begin(self, model):
        raise NotImplementedError

    def on_bu_pred_begin(self, model):
        return True

    def on_td_begin(self, model, img_features, last_bu, bu_out, td_out):
        return None

    def on_init_end(self, model):
        pass

    def on_forward_end(self, bu_out, td_out):
        return bu_out, td_out


class RecurrentInstructor(BaseInstructor):
    def __init__(self, repeats):
        self.repeats = repeats
        super().__init__()

    def on_bu_body_begin(self, model):
        should_continue = self.i < self.repeats
        return should_continue
