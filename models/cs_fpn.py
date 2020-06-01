import torch
from torch import nn
import torch.nn.functional as F
from fastai.callbacks import model_sizes
from fastai.layers import conv2d

from models import layers


class SplitHead(nn.Module):

    def __init__(self, in_dim, out_dims):
        super().__init__()
        d = {key: nn.Sequential(layers.conv_layer(in_dim, in_dim), conv2d(in_dim, fn, ks=1, bias=True))
             for key, fn in out_dims.items()}
        self.heads = nn.ModuleDict(d)

    def forward(self, x):
        out = {k: m(x) for k, m in self.heads.items()}
        return out


class BottomUp(nn.ModuleList):

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, x):
        out = [self[0](x)]
        for layer in self[1:]:
            out.append(layer(out[-1]))
        return out


class BottomUpWithLaterals(nn.Module):
    def __init__(self, modules, channels_out, channels_in):
        super().__init__()
        self.bb = nn.ModuleList(modules)
        laterals = [layers.conv_layer(c_in, c_out) for c_in, c_out in zip(channels_in, channels_out)]
        self.laterals = nn.ModuleList(laterals)

    def _layer_forward(self, x, i, laterals_in):
        out = self.bb[i](x)
        if laterals_in:
            out = out + self.laterals[i](laterals_in[i])
        return out

    def forward(self, x, lateral_in=None):
        out = [self._layer_forward(x, 0, lateral_in)]
        for i in range(1, len(self.bb) - 1):
            out.append(self._layer_forward(out[-1], i, lateral_in))
        out.append(self.bb[-1](out[-1]))
        return out


class TopDown(nn.Module):

    def __init__(self, channels, dim):
        super().__init__()
        td_in, td_out = [], []
        for c in reversed(channels):
            td_in.append(layers.conv_layer(c, dim, ks=1))
            td_out.append(layers.conv_layer(dim, dim, ks=3))
        self.td_in = nn.ModuleList(td_in)
        self.td_out = nn.ModuleList(td_out)

    def forward(self, x):
        out = []
        last_inner = self.td_in[0](x[0])
        out.append(self.td_out[0](last_inner))

        for lateral_layer, out_layer, lateral_in in zip(self.td_in[1:], self.td_out[1:], x[1:]):
            lateral_f = lateral_layer(lateral_in)
            feat_shape = lateral_f.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="bilinear", align_corners=False)
            last_inner = lateral_f + inner_top_down
            out.append(out_layer(last_inner))
        return out


class Fusion(nn.Module):

    def __init__(self, n, dim):
        super().__init__()
        self.conv = layers.conv_layer(n * dim, dim, ks=3)

    def forward(self, x):
        out_size = x[0].shape[-2:]
        fusion_list = [x[0]]
        for layer_f in x[1:]:
            fusion_list.append(F.interpolate(layer_f, size=out_size, mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        out = self.conv(fusion_out)
        return out


def build_fpn(body, fpn_dim, bu_in_lateral=False, out_dim=None):
    layers_idx = 4
    ifn = body[:layers_idx]
    bu = body[layers_idx:]
    szs = model_sizes(bu)
    channels = [s[1] for s in szs]

    bu_bb = [m for m in bu]
    if not bu_in_lateral:
        bu = BottomUp(bu_bb)
    else:
        channel_in = [out_dim] + [fpn_dim, fpn_dim]
        channels_out = channels[:3]
        bu = BottomUpWithLaterals(bu_bb, channels_out, channel_in)
    td = TopDown(channels, fpn_dim)

    fusion = Fusion(len(channels), fpn_dim)
    return ifn, bu, td, fusion, channels


class FPN(nn.Module):

    def __init__(self, body, out_dims, fpn_dim=256):
        super().__init__()
        self.ifn, self.bu, self.td, self.fusion, _ = build_fpn(body, fpn_dim)
        self.td_head = SplitHead(fpn_dim, out_dims)

    def forward(self, images):
        features = self.ifn(images)
        bu_out = self.bu(features)
        bu_out.reverse()

        td_out = self.td(bu_out)
        td_out.reverse()

        out = self.fusion(td_out)
        out = self.td_head(out)
        return out


class TwoIterFPN(nn.Module):

    def __init__(self, body, out_dims, fpn_dim=256):
        super().__init__()
        self.ifn, self.bu, self.td, self.fusion, ch = build_fpn(body, fpn_dim, bu_in_lateral=True,
                                                                out_dim=out_dims['object'])
        c = ch[-1]
        embeddings = {k: nn.Sequential(layers.conv_layer(c, 2 * c), layers.conv_layer(2 * c, c))
                      for k in out_dims.keys()}
        self.embedding = nn.ModuleDict(embeddings)
        head = {key: nn.Sequential(layers.conv_layer(fpn_dim, fpn_dim), conv2d(fpn_dim, fn, ks=1, bias=True))
                for key, fn in out_dims.items()}
        self.head = nn.ModuleDict(head)

    def forward(self, images):
        out = {}

        features = self.ifn(images)
        bu_out = self.bu(features)
        bu_out[-1] = self.embedding['object'](bu_out[-1])
        bu_out.reverse()

        td_out = self.td(bu_out)
        out['object'] = self.head['object'](self.fusion(td_out))

        bu_lateral_in = [out['object']] + td_out[-2:0:-1]
        bu_out = self.bu(features, lateral_in=bu_lateral_in)
        bu_out[-1] = self.embedding['part'](bu_out[-1])
        bu_out.reverse()

        td_out = self.td(bu_out)
        out['part'] = self.head['part'](self.fusion(td_out))
        return out
