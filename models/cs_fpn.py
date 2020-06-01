import torch
from torch import nn
import torch.nn.functional as F
from fastai.callbacks import model_sizes
from fastai.layers import conv2d

from models import layers


def build_fpn(body, fpn_dim):
    layers_idx = 4
    ifn = body[:layers_idx]
    bu = body[layers_idx:]
    szs = model_sizes(bu)
    channels = [s[1] for s in szs]
    bu = nn.ModuleList([m for m in bu])

    td_in, td_out = [], []
    for c in reversed(channels):
        td_in.append(layers.conv_layer(c, fpn_dim, ks=1))
        td_out.append(layers.conv_layer(fpn_dim, fpn_dim, ks=3))
    td_in = nn.ModuleList(td_in)
    td_out = nn.ModuleList(td_out)

    td_bu_laterals = []
    for c in channels:
        td_bu_laterals.append(layers.conv_layer(fpn_dim, c))
    td_bu_laterals = nn.ModuleList(td_bu_laterals)

    fusion = layers.conv_layer(len(channels) * fpn_dim, fpn_dim, ks=1)
    return ifn, bu, td_in, td_out, td_bu_laterals, fusion


class FPN(nn.Module):

    def __init__(self, body, out_dims, fpn_dim=256):
        super().__init__()
        self.ifn, self.bu, self.td_in, self.td_out, self.td_bu_laterals, self.fusion = build_fpn(body, fpn_dim)
        self.td_head = SplitHead(fpn_dim, out_dims)

    def forward(self, images):
        features = self.ifn(images)
        bu_out = [self.bu[0](features)]
        for i in range(1, len(self.bu)):
            layer = self.bu[i]
            bu_out.append(layer(bu_out[-1]))

        bu_out.reverse()
        td_out = []
        last_inner = self.td_in[0](bu_out[0])
        td_out.append(self.td_out[0](last_inner))

        for i in range(1, len(self.td_out)):
            lateral = self.td_in[i](bu_out[i])
            feat_shape = lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="bilinear", align_corners=False)
            last_inner = lateral + inner_top_down
            td_out.append(self.td_out[i](last_inner))

        td_out.reverse()
        out_size = td_out[0].shpae[-2:]
        fusion_list = [td_out[0]]
        for i in range(1, len(td_out)):
            fusion_list.append(F.interpolate(td_out[i], size=out_size, mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        out = self.fusion(fusion_out)
        out = self.td_head(out)
        return out


class SplitHead(nn.Module):

    def __init__(self, in_dim, out_dims):
        super().__init__()
        d = {}
        for key, fn in out_dims.items():
            d[key] = nn.Sequential(layers.conv_layer(in_dim, in_dim), conv2d(in_dim, fn, ks=1, bias=True))
        self.heads = nn.ModuleDict(d)

    def forward(self, x):
        out = {k: m(x) for k, m in self.heads.items()}
        return out
