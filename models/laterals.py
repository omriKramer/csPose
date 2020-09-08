import torch
from fastai import vision as fv
from torch import nn
from torch.nn import functional as F

from models.nnlayers import conv_layer, conv1d


class Lateral(nn.Module):
    def __init__(self, origin_layer: nn.Module, target_layer: nn.Module, detach=False):
        super().__init__()
        self.detach = detach
        self.origin_out = None
        self.origin_hook = origin_layer.register_forward_hook(self.origin_forward_hook)
        self.target_hook = target_layer.register_forward_pre_hook(self.target_forward_pre_hook)

    def origin_forward_hook(self, module, inp, output):
        if self.detach:
            output = output.detach()
        self.origin_out = output

    def target_forward_pre_hook(self, module, inp):
        result = self(self.origin_out, inp[0])
        if result is not None:
            return (result, *inp[1:])
        return None

    def remove(self):
        self.origin_hook.remove()
        self.target_hook.remove()


class ConvAddLateral(Lateral):
    def __init__(self, origin_layer, target_layer, channels, ks=3, detach=False):
        super().__init__(origin_layer, target_layer, detach)
        self.conv = conv_layer(channels, channels, ks=ks)

    def forward(self, origin_out, target_input):
        if origin_out is None:
            return

        out = self.conv(origin_out)
        out = out + target_input
        return out


class LateralSoftAdd(Lateral):
    def __init__(self, origin_layer, target_layer, channels, detach=False):
        super().__init__(origin_layer, target_layer, detach)
        self.alpha = nn.Parameter(torch.zeros(channels), requires_grad=True)

    def forward(self, origin_out, target_input):
        if origin_out is None:
            return

        return target_input + self.alpha[:, None, None] * origin_out


class LateralConvMul(Lateral):
    def __init__(self, origin_layer, target_layer, channels, ks, detach=False):
        super().__init__(origin_layer, target_layer, detach=detach)
        self.conv = conv_layer(channels, channels, ks=ks)

    def forward(self, origin_out, target_input):
        if origin_out is None:
            return

        out = self.conv(origin_out)
        out = out * target_input
        return out


class AttentionLateralOp(Lateral):

    def __init__(self, origin_layer, target_layer, n_channels: int):
        super().__init__(origin_layer, target_layer)
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(fv.tensor([0.]), requires_grad=True)

    def forward(self, origin_out, target_in):
        if origin_out is None:
            return

        size = origin_out.size()
        origin_out = origin_out.view(*size[:2], -1)
        target_in = target_in.view(*size[:2], -1)

        f, g, h = self.query(target_in), self.key(origin_out), self.value(origin_out)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + target_in
        return o.view(*size).contiguous()


def create_laterals(lateral, origin_net, target_net, channels, **kwargs):
    laterals = [lateral(o_layer, t_layer, c, **kwargs)
                for o_layer, t_layer, c in zip(origin_net, reversed(target_net), channels)]
    return nn.ModuleList(laterals)


class HeatmapAdd(Lateral):
    def __init__(self, origin_layer, target_layer, ni, nf):
        super().__init__(origin_layer, target_layer)
        self.bn = nn.BatchNorm2d(ni)
        self.conv = conv_layer(ni, nf)

    def forward(self, origin_out, target_in):
        if origin_out is None:
            return

        out = self.bn(origin_out)
        out = self.conv(out)
        out = out + target_in
        return out


class DenseLateral(Lateral):

    def __init__(self, origin_layer, target_layer):
        super().__init__(origin_layer, target_layer)

    def forward(self, origin_out, target_in):
        if origin_out is None:
            origin_out = torch.zeros_like(target_in)

        return torch.cat((target_in, origin_out), dim=1)
