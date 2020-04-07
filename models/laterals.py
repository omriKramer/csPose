import torch
from fastai import vision as fv
from torch import nn
from torch.nn import functional as F

from models.layers import conv_layer, conv1d


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


class LateralSoftAddOp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels), requires_grad=True)

    def forward(self, origin_out, target_input):
        return target_input + self.alpha[:, None, None] * origin_out


def soft_add_lateral(origin_layer, target_layer, channels):
    return Lateral(origin_layer, target_layer, LateralSoftAddOp(channels))


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


class HeatmapAddOp(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv = conv_layer(ni, nf)

    def forward(self, origin_out, target_in):
        out = self.bn(origin_out)
        out = self.conv(out)
        out = out + target_in
        return out


def heatmap_add_lateral(origin_layer, target_layer, ni, nf, detach=True):
    op = HeatmapAddOp(ni, nf)
    return Lateral(origin_layer, target_layer, op, detach=detach)
