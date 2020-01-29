from typing import Callable, Tuple, Any, Union, List

import fastai.vision as fv
import numpy as np
import torch
import torch.nn.functional as F
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


class Lateral(nn.Module):
    def __init__(self, origin_layer: nn.Module, target_layer: nn.Module, channels: int, detach=False):
        super().__init__()
        self.detach = detach
        self.origin_out = None
        self.lateral = nn.Conv2d(channels, channels, 1, bias=False)
        self.bu_hook = origin_layer.register_forward_hook(self.origin_forward_hook)
        self.td_hook = target_layer.register_forward_pre_hook(lambda module, inp: self(inp[0]))

    def origin_forward_hook(self, module, inp, output):
        if self.detach:
            output = output.detach()
        self.origin_out = output

    def forward(self, inp):
        if self.origin_out is not None:
            out = self.lateral(self.origin_out) + inp
            return out


def create_laterals(origin_net, target_net, channels, **kwargs):
    laterals = [Lateral(o_layer, t_layer, c, **kwargs)
                for o_layer, t_layer, c in zip(origin_net, reversed(target_net), channels)]
    return nn.ModuleList(laterals)


class TDBlock(nn.Module):

    def __init__(self, c_in, c_out, upsample=False):
        super().__init__()
        self.conv1 = fv.conv_layer(c_in, c_in)
        self.conv2 = fv.conv_layer(c_in, c_out)
        self.upsample = upsample

    def forward(self, x):
        out = self.conv1(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv2(out)
        return out


class TDHead(nn.Sequential):
    def __init__(self, fi, fn):
        super().__init__(fv.conv_layer(fi, fi),
                         fv.conv2d(fi, fn, ks=1))


class CounterStream(nn.Module):
    def __init__(self, bu, instructor, td_c=1, bu_c=0, detach=False, td_laterals=True, embedding=fv.embedding,
                 img_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        # first few layers are not in a block, we convert all layers up through MaxPool to a single block
        concat_idx = 4
        self.bu_body = nn.Sequential(bu[:concat_idx], *bu[concat_idx:])
        szs = fv.learner.model_sizes(self.bu_body, size=img_size)

        td = []
        for inp_size, out_size in zip(reversed(szs), reversed(szs[:-1])):
            upsample = inp_size[-1] != out_size[-1]
            td.append(TDBlock(inp_size[1], out_size[1], upsample=upsample))
        channels = [s[1] for s in szs]
        self.td = nn.Sequential(*td, TDHead(channels[0], td_c))

        self.laterals = create_laterals(self.bu_body[1:], self.td[:-1], channels[1:], detach=detach)
        if td_laterals:
            self.laterals.extend(
                create_laterals(self.td[:-1], self.bu_body[1:], reversed(channels[:-1]), detach=detach))

        self.emb = embedding(instructor.n_inst, channels[-1]) if embedding else None
        self.bu_head = fv.create_head(channels[-1] * 2, bu_c) if bu_c else None
        self.instructor = instructor

    def clear(self):
        for lateral in self.laterals:
            lateral.origin_out = None

    def forward(self, img):
        self.clear()
        should_continue = True
        td_out, bu_out = [], []

        while should_continue:
            current_bu = self.bu_body(img)
            bu_shape = current_bu.shape
            if self.bu_head:
                current_bu = self.bu_head(current_bu)

            inst, should_continue = self.instructor.next_inst(current_bu)
            inst_emb = self.emb(inst)[..., None, None] if self.emb else current_bu.new_zeros(1)
            inst_emb = inst_emb.expand(bu_shape)
            current_td = self.td(inst_emb)

            bu_out.append(current_bu)
            td_out.append(current_td)

        return torch.cat(bu_out, dim=1), torch.cat(td_out, dim=1)


def cs_learner(data: fv.DataBunch, arch: Callable, td_c, instructor, bu_c=0, td_laterals=True, embedding=fv.embedding,
               pretrained: bool = True, cut: Union[int, Callable] = None, **learn_kwargs: Any) -> fv.Learner:
    """Build Counter Stream learner from `data` and `arch`."""
    body = fv.create_body(arch, pretrained, cut)
    size = next(iter(data.train_dl))[0].shape[-2:]
    model = fv.to_device(
        CounterStream(body, instructor, td_c=td_c, bu_c=bu_c, img_size=size,
                      td_laterals=td_laterals, embedding=embedding),
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


class SequentialInstructor(BaseInstructor):
    def __init__(self, instructions):
        self.instructions = torch.tensor(instructions)
        self.reindex = self.instructions.argsort()

    def on_batch_begin(self, last_input, last_output, last_target, train, **kwargs):
        batch_size = last_input.shape[0]
        instructions = self.instructions.to(device=last_input.device).expand(batch_size, len(self.instructions)).T
        return {'last_input': (last_input, instructions)}

    def on_loss_begin(self, last_input, last_output, last_target, train, **kwargs: Any):
        bu_out, td_out = last_output
        bu_out, td_out = bu_out[:, self.reindex], td_out[:, self.reindex]
        return {'last_target': (bu_out, td_out)}

    @property
    def n_inst(self):
        return len(self.instructions)


class SingleInstruction(BaseInstructor):
    n_inst = 1

    def next_inst(self, bu_out):
        batch_size = bu_out.shape[0]
        instructions = torch.zeros(batch_size, dtype=torch.long, device=bu_out.device)
        return instructions, False
