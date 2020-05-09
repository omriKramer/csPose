import itertools
from abc import ABC
from typing import Callable, Tuple, Any, List

import fastai.vision as fv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from . import laterals, layers


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


def _group_td(td, bu):
    """group TDBlocks to mirror the layer groups in the BU network"""
    layer_len = [len(layer) for layer in bu]
    layer_len.reverse()
    end_idx = np.cumsum(layer_len)
    start_idx = np.roll(end_idx, 1)
    start_idx[0] = 0
    td_layered = []
    for start, end in zip(start_idx, end_idx):
        td_layered.append(nn.Sequential(*td[start:end]))
    return td_layered


class TDBlock(nn.Module):

    def __init__(self, upsample, mode='bilinear'):
        super().__init__()
        self.upsample = upsample
        self.mode = mode

    def maybe_upsample(self, x, identity):
        if self.upsample:
            identity = self.upsample(identity)
            if x.shape[-1] != identity.shape[-1]:
                x = F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=False)

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
        super().__init__(layers.conv_layer(fi, fi),
                         nn.Conv2d(fi, fn, kernel_size=1, bias=True))


class UnetBlock(nn.Module):

    def __init__(self, ni, nf, upsample=False):
        super().__init__()
        self.conv1 = conv3x3(ni, ni)
        self.bn1 = nn.BatchNorm2d(ni)
        self.conv2 = conv3x3(ni, nf)
        self.bn2 = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.relu(self.bn2(self.conv2(out)))
        return out


def _bu_laterals_idx(bu):
    lengths = [len(layer) for layer in bu]
    lengths.reverse()
    idx = [0] + list(np.cumsum(lengths))[:-1]
    return set(idx)


class DoubleUnet(nn.Module):

    def __init__(self, bu, iterations=2, td_c=16, img_size=(256, 256), weighted_sum=False):
        super().__init__()
        concat_idx = 4
        self.fe = bu[:concat_idx]
        self.bu = nn.Sequential(*bu[concat_idx:])
        self.iterations = iterations
        self.weights = torch.ones()

        bu_flat = [bu[:concat_idx]] + list(itertools.chain(*bu[concat_idx:]))
        bu_flat = nn.Sequential(*bu_flat)
        szs = fv.learner.model_sizes(bu_flat, img_size)
        ni = szs[-1][1]
        self.middle_conv = nn.Sequential(
            layers.conv_layer(ni, ni * 2),
            layers.conv_layer(ni * 2, ni)
        )

        szs.reverse()
        td = []
        lat_idx = _bu_laterals_idx(self.bu)
        for i, (szs_in, szs_out) in enumerate(zip(szs, szs[1:])):
            c_in = szs_in[1]
            if i in lat_idx:
                c_in *= 2

            upsample = szs_in[-1] != szs_out[-1]
            td.append(UnetBlock(c_in, szs_out[1], upsample=upsample))

        self.td = nn.Sequential(*_group_td(td, self.bu))
        c = szs[-1][1]
        self.td_head = nn.Sequential(
            layers.conv_layer(c, c),
            conv1x1(c, td_c)
        )

        for layer in self.bu:
            double_res_block(layer[0])

        self.bu_laterals = []
        self.td_laterals = []
        for bu_l, td_l in zip(self.bu, self.td[::-1]):
            self.bu_laterals.append(laterals.DenseLateral(bu_l, td_l))
            self.td_laterals.append(laterals.DenseLateral(td_l, bu_l))

    def clear(self):
        for lateral in itertools.chain(self.bu_laterals, self.td_laterals):
            del lateral.origin_out
            lateral.origin_out = None

    def forward(self, img):
        img_features = self.fe(img)
        out = []
        for _ in range(self.iterations):
            x = self.bu(img_features)
            x = self.middle_conv(x)
            x = self.td(x)
            out.append(self.td_head(x))

        self.clear()
        return out


def double_unet_learner(data: fv.DataBunch, arch: Callable, iterations=2, td_c=16,
                        **learn_kwargs: Any) -> fv.Learner:
    """Build Counter Stream learner from `data` and `arch`."""
    body = fv.create_body(arch, pretrained=False)
    size = next(iter(data.train_dl))[0].shape[-2:]
    model = DoubleUnet(body, iterations=iterations, td_c=td_c, img_size=size)
    model = fv.to_device(model, data.device)
    learn = fv.Learner(data, model, **learn_kwargs)
    fv.apply_init(learn.model, nn.init.kaiming_normal_)
    return learn


def double_conv(conv):
    in_c, out_c, ks, s, p = conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding
    return nn.Conv2d(in_c * 2, out_c, kernel_size=ks, stride=s, padding=p, bias=False)


def double_res_block(block):
    block.conv1 = double_conv(block.conv1)
    if block.downsample:
        block.downsample[0] = double_conv(block.downsample[0])
    else:
        c_in = block.conv1.in_channels
        try:
            c_out = block.bn3.num_features
        except AttributeError:
            c_out = block.bn2.num_features

        if c_in != c_out:
            block.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out)
            )


class Fuse(nn.Module):

    def __init__(self, m, channels, out_c):
        super().__init__()
        self.hooks = fv.callbacks.hook_outputs(m, detach=False)
        self.fuse_conv = layers.conv_layer(sum(channels), out_c)

    def forward(self, x):
        size = x.shape[-2:]
        stored = [F.interpolate(o, size=size, mode='bilinear', align_corners=False) for o in self.hooks.stored]
        x = torch.cat([*stored, x], dim=1)
        out = self.fuse_conv(x)
        return out


class CounterStream(nn.Module):

    def __init__(self, bu, instructor, td_c=1, bu_c=0, lateral=laterals.ConvAddLateral,
                 td_out_lateral=None, embedding=fv.embedding, lateral_on='block', ppm=False, fuse=False,
                 img_size: Tuple[int, int] = (256, 256), bu_lateral=None, td_lateral=None):
        super().__init__()
        # first few layers are not in a block, we group all layers up through MaxPool to a single block
        concat_idx = 4
        self.ifn = bu[:concat_idx]
        self.bu_body = nn.Sequential(*bu[concat_idx:])
        bu = [bu[:concat_idx]] + list(itertools.chain(*bu[concat_idx:]))
        bu = nn.Sequential(*bu)

        if isinstance(bu[1], torchvision.models.resnet.BasicBlock):
            td_block = TDBasicBlock
        elif isinstance(bu[1], torchvision.models.resnet.Bottleneck):
            td_block = TDBottleNeck
        else:
            raise ValueError

        szs = fv.learner.model_sizes(bu, img_size)
        td_szs = list(reversed(szs))
        td = []
        for sz_in, sz_out in zip(td_szs, td_szs[1:]):
            upsample = None
            if sz_in[-1] != sz_out[-1]:
                upsample = nn.Sequential(
                    fv.Lambda(lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)),
                    fv.conv_layer(sz_in[1], sz_out[1], ks=1, use_activ=False)
                )
            elif sz_in[1] != sz_out[1]:
                upsample = fv.conv_layer(sz_in[1], sz_out[1], ks=1, use_activ=False)
            td.append(td_block(sz_in[1], sz_out[1], upsample=upsample))

        td_layered = _group_td(td, self.bu_body)
        channels = [s[1] for s in szs]

        td_head = TDHead(channels[0], td_c)
        if fuse:
            td_channels = list(reversed(channels))
            layers_c, i = [], 0
            for layer in td_layered:
                i += len(layer)
                layers_c.append(td_channels[i])

            fuse = Fuse(td_layered[:-1], layers_c, channels[0])
            td_head = nn.Sequential(fuse, td_head)

        self.td = nn.Sequential(*td_layered, td_head)

        td.append(self.td[-1])

        if not bu_lateral:
            bu_lateral = lateral
        if not td_lateral:
            td_lateral = lateral
        if lateral_on == 'layer':
            bu = [self.ifn] + [layer for layer in self.bu_body]
            td = self.td

        self.bu_laterals = laterals.create_laterals(bu_lateral, bu[:-1], td[1:], channels[:-1])
        self.td_laterals = laterals.create_laterals(td_lateral, td[:-1], bu[1:], reversed(channels[:-1]))

        if td_out_lateral:
            self.td_laterals[-1].remove()
            hm_lat = td_out_lateral(self.td[-1], self.bu_body[0], td_c, channels[0])
            self.td_laterals[-1] = hm_lat

        if ppm:
            self.td = nn.Sequential(layers.PPM(channels[-1]), *self.td)

        self.emb = embedding(instructor.n_inst, channels[-1]) if embedding else None
        self.bu_head = fv.create_head(channels[-1] * 2, bu_c) if bu_c else None
        self.instructor = instructor
        self.instructor.on_init_end(self)

    def clear(self):
        for lateral in itertools.chain(self.bu_laterals, self.td_laterals):
            del lateral.origin_out
            lateral.origin_out = None

    def forward(self, img):
        self.instructor.on_forward_begin(self)
        img_features = self.ifn(img)

        td_out, bu_out = [], []
        while self.instructor.on_iter_begin(self):

            if self.instructor.on_bu_begin(self):
                last_bu = self.bu_body(img_features)
                if self.bu_head:
                    bu_out.append(self.bu_head(last_bu))

            inst = self.instructor.on_td_begin(self, img_features, last_bu, bu_out, td_out)
            if self.emb:
                last_bu = last_bu * self.emb(inst)[..., None, None]
            td_out.append(self.td(last_bu))

            self.instructor.i += 1

        bu_out = torch.cat(bu_out, dim=1) if bu_out else None
        td_out = torch.cat(td_out, dim=1)
        self.clear()
        return self.instructor.on_forward_end(bu_out, td_out)


def cs_learner(data: fv.DataBunch, arch: Callable, instructor, td_c=1, bu_c=0, embedding=fv.embedding,
               lateral=laterals.ConvAddLateral, td_out_lateral=None, ppm=False,
               pretrained: bool = True, **learn_kwargs: Any) -> fv.Learner:
    """Build Counter Stream learner from `data` and `arch`."""
    body = fv.create_body(arch, pretrained)
    size = next(iter(data.train_dl))[0].shape[-2:]
    model = fv.to_device(
        CounterStream(body, instructor, td_c=td_c, bu_c=bu_c, img_size=size, embedding=embedding,
                      lateral=lateral, td_out_lateral=td_out_lateral, ppm=ppm),
        data.device)
    learn = fv.Learner(data, model, **learn_kwargs)
    learn.split([learn.model.td[0]])
    if pretrained:
        learn.freeze()
        fv.apply_init(learn.model.bu_laterals, nn.init.kaiming_normal_)
        fv.apply_init(learn.model.td_laterals, nn.init.kaiming_normal_)
        if learn.model.bu_head:
            fv.apply_init(learn.model.bu_head, nn.init.kaiming_normal_)
    else:
        fv.apply_init(learn.model, nn.init.kaiming_normal_)
    return learn


class BaseInstructor(ABC):
    def __init__(self):
        self.i = 0

    def on_forward_begin(self, model):
        self.i = 0

    def on_iter_begin(self, model):
        raise NotImplementedError

    def on_bu_begin(self, model):
        return True

    def on_td_begin(self, model, img_features, last_bu, bu_out, td_out):
        return None

    def on_init_end(self, model):
        pass

    def on_forward_end(self, bu_out, td_out):
        if bu_out is None:
            return td_out

        return bu_out, td_out


class RecurrentInstructor(BaseInstructor):
    def __init__(self, repeats):
        self.repeats = repeats
        super().__init__()

    def on_iter_begin(self, model):
        should_continue = self.i < self.repeats
        return should_continue
