import itertools
from typing import Callable, Tuple, Any, Union, List

import fastai.vision as fv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn


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


def conv_layer(ni, nf, ks=3):
    return nn.Sequential(
        fv.conv2d(ni, nf, ks=ks, bias=False),
        fv.batchnorm_2d(nf),
        nn.ReLU(inplace=True)
    )


class Lateral(nn.Module):
    def __init__(self, origin_layer: nn.Module, target_layer: nn.Module, op, detach=False):
        super().__init__()
        self.detach = detach
        self.origin_out = None
        self.origin_hook = origin_layer.register_forward_hook(self.origin_forward_hook)
        self.target_hook = target_layer.register_forward_pre_hook(lambda module, inp: self(inp[0]))
        self.op = op

    def origin_forward_hook(self, module, inp, output):
        if self.detach:
            output = output.detach()
        self.origin_out = output

    def forward(self, inp):
        if self.origin_out is None:
            return

        out = self.op(self.origin_out, inp)
        return out


class LateralConvAddOp(nn.Module):
    def __init__(self, channels, ks):
        super().__init__()
        self.conv = conv_layer(channels, channels, ks=ks)

    def forward(self, origin_out, target_input):
        out = self.conv(origin_out)
        out = out + target_input
        return out


def conv_add_lateral(origin_layer, target_layer, channels, detach=False, ks=3):
    op = LateralConvAddOp(channels, ks)
    return Lateral(origin_layer, target_layer, op, detach=detach)


class LateralConvMulOP(nn.Module):
    def __init__(self, channels, ks):
        super().__init__()
        self.conv = conv_layer(channels, channels, ks=ks)

    def forward(self, origin_out, target_input):
        out = self.conv(origin_out)
        out = out * target_input
        return out


def conv_mul_lateral(origin_layer, target_layer, channels, detach=False, ks=3):
    op = LateralConvMulOP(channels, ks)
    return Lateral(origin_layer, target_layer, op, detach=detach)


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """Create and initialize a `nn.Conv1d` layer with spectral normalization."""
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return fv.spectral_norm(conv)


class AttentionLateralOp(nn.Module):

    def __init__(self, n_channels: int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(fv.tensor([0.]), requires_grad=True)

    def forward(self, origin_out, target_in):
        size = origin_out.size()
        origin_out = origin_out.view(*size[:2], -1)
        target_in = target_in.view(*size[:2], -1)

        f, g, h = self.query(target_in), self.key(origin_out), self.value(origin_out)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + target_in
        return o.view(*size).contiguous()


def attention_lateral(origin_layer, target_layer, channels, detach=False):
    op = AttentionLateralOp(channels)
    return Lateral(origin_layer, target_layer, op, detach=detach)


def create_laterals(lateral, origin_net, target_net, channels, **kwargs):
    laterals = [lateral(o_layer, t_layer, c, **kwargs)
                for o_layer, t_layer, c in zip(origin_net, reversed(target_net), channels)]
    return nn.ModuleList(laterals)


class TDBlock(nn.Module):
    def __init__(self, *layers, upsample=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x):
        identity = x
        out = self.layers[0](x)
        if self.upsample:
            identity = self.upsample(identity)
            if out.shape[-1] != identity.shape[-1]:
                out = F.interpolate(out, scale_factor=2, mode='nearest')

        for layer in self.layers[1:]:
            out = layer(out)
        out += identity
        out = self.relu(out)
        return out


class TDBasicBlock(TDBlock):

    def __init__(self, ni, nf, upsample=None):
        super().__init__(
            fv.conv_layer(ni, ni),
            fv.conv_layer(ni, nf),
            fv.batchnorm_2d(nf),
            upsample=upsample)


class TDBottleNeck(TDBlock):
    expansion = 4

    def __init__(self, ni, nf, upsample=None):
        width = ni // self.expansion
        super().__init__(
            fv.conv_layer(ni, width),
            fv.conv_layer(width, width),
            fv.conv2d(width, nf),
            fv.batchnorm_2d(nf),
            upsample=upsample
        )


class TDHead(nn.Sequential):
    def __init__(self, fi, fn):
        super().__init__(fv.conv_layer(fi, fi),
                         fv.conv2d(fi, fn, ks=1))


class CounterStream(nn.Module):
    def __init__(self, bu, instructor, td_c=1, bu_c=0, detach=False, td_detach=None, td_laterals=True,
                 embedding=fv.embedding, img_size: Tuple[int, int] = (256, 256), lateral=conv_add_lateral):
        super().__init__()
        # first few layers are not in a block, we group all layers up through MaxPool to a single block
        concat_idx = 4
        self.bu_body = nn.Sequential(bu[:concat_idx], *bu[concat_idx:])
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
                    fv.conv_layer(sz_in[1], sz_out[1], use_activ=False)
                )
            elif sz_in[1] != sz_out[1]:
                upsample = fv.conv_layer(sz_in[1], sz_out[1], ks=1, use_activ=False)
            td.append(td_block(sz_in[1], sz_out[1], upsample=upsample))

        td_layered = self._group_td(td)
        self.td = nn.Sequential(*td_layered, TDHead(td_szs[-1][1], td_c))

        channels = [s[1] for s in szs]
        td.append(self.td[-1])
        self.laterals = create_laterals(lateral, bu[:-1], td[1:], channels[:-1], detach=detach)
        if td_laterals:
            td_detach = td_detach if td_detach is not None else detach
            bu_laterals = create_laterals(lateral, td[:-1], bu[1:], reversed(channels[:-1]), detach=td_detach)
            self.laterals.extend(bu_laterals)

        self.emb = embedding(instructor.n_inst, channels[-1]) if embedding else None
        self.bu_head = fv.create_head(channels[-1] * 2, bu_c) if bu_c else None
        self.instructor = instructor

    def _group_td(self, td):
        """group TDBlocks to mirror the layer groups in the BU network"""
        layer_len = [len(layer) for layer in self.bu_body[1:]]
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
        state = {'clear': True}
        td_out, bu_out = [], []

        while state.get('continue', True):
            if state.get('clear', False):
                self.clear()

            last_bu = self.bu_body(img)
            if self.bu_head:
                bu_out.append(self.bu_head(last_bu))

            inst, state = self.instructor.next_inst(bu_out[-1] if bu_out else last_bu)
            if self.emb:
                last_bu = last_bu * self.emb(inst)[..., None, None]
            td_out.append(self.td(last_bu))

        self.clear()
        bu_out = torch.cat(bu_out, dim=1) if bu_out else None
        td_out = torch.cat(td_out, dim=1)
        return bu_out, td_out


def cs_learner(data: fv.DataBunch, arch: Callable, instructor, td_c=1, bu_c=0, td_laterals=True, embedding=fv.embedding,
               detach=False, td_detach=None, lateral=conv_add_lateral,
               pretrained: bool = True, cut: Union[int, Callable] = None, **learn_kwargs: Any) -> fv.Learner:
    """Build Counter Stream learner from `data` and `arch`."""
    body = fv.create_body(arch, pretrained, cut)
    size = next(iter(data.train_dl))[0].shape[-2:]
    model = fv.to_device(
        CounterStream(body, instructor, td_c=td_c, bu_c=bu_c, img_size=size, embedding=embedding,
                      td_laterals=td_laterals, detach=detach, td_detach=td_detach, lateral=lateral),
        data.device)
    learn = fv.Learner(data, model, callbacks=instructor, **learn_kwargs)
    learn.split((learn.model.bu_body[3], learn.model.td[0]))
    if pretrained:
        learn.freeze()
    return learn


class BaseInstructor(fv.Callback):
    _order = 20

    def next_inst(self, bu_out):
        raise NotImplementedError


class SingleInstruction(BaseInstructor):
    n_inst = 1

    def __init__(self):
        self.state = {'continue': False}

    def next_inst(self, bu_out):
        batch_size = bu_out.shape[0]
        instructions = torch.zeros(batch_size, dtype=torch.long, device=bu_out.device)
        return instructions, self.state


class RecurrentInstructor(BaseInstructor):
    def __init__(self, repeats):
        self.repeats = repeats
        self.i = 0

    def on_batch_begin(self, **kwargs):
        self.i = 0

    def next_inst(self, last_bu):
        self.i += 1
        state = {'continue': self.i < self.repeats}
        return None, state
