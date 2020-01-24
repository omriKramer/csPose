import itertools
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
    def __init__(self, bu, n_instructions, c_out=1, detach=False, img_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        # first few layers are not in a block, we convert all layers up through MaxPool to a single block
        concat_idx = 4
        self.bu = nn.Sequential(bu[:concat_idx], *bu[concat_idx:])
        szs = fv.learner.model_sizes(self.bu, size=img_size)

        td = []
        for inp_size, out_size in zip(reversed(szs), reversed(szs[:-1])):
            upsample = inp_size[-1] != out_size[-1]
            td.append(TDBlock(inp_size[1], out_size[1], upsample=upsample))
        channels = [s[1] for s in szs]
        self.td = nn.Sequential(*td, TDHead(channels[0], c_out))

        self.bu_laterals = create_laterals(self.bu[1:], self.td[:-1], channels[1:], detach=detach)
        self.td_laterals = create_laterals(self.td[:-1], self.bu[1:], reversed(channels[:-1]), detach=detach)
        self.emb = fv.embedding(n_instructions, channels[-1])

    def clear(self):
        for lateral in itertools.chain(self.bu_laterals, self.td_laterals):
            lateral.origin_out = None

    def forward(self, img, instructions):
        td_out, bu_out = [], []
        for inst in instructions:
            current_bu = self.bu(img)
            inst_emb = self.emb(inst)[..., None, None].expand_as(current_bu)
            current_td = self.td(inst_emb)

            bu_out.append(current_bu)
            td_out.append(current_td)

        return torch.cat(bu_out, dim=1), torch.cat(td_out, dim=1)


def cs_learner(data: fv.DataBunch, arch: Callable, instructions, pretrained: bool = True, c_out=1,
               cut: Union[int, Callable] = None, **learn_kwargs: Any) -> fv.Learner:
    """Build Counter Stream learner from `data` and `arch`."""
    body = fv.create_body(arch, pretrained, cut)
    size = next(iter(data.train_dl))[0].shape[-2:]
    model = fv.to_device(CounterStream(body, len(instructions), c_out=c_out, img_size=size), data.device)
    learn = fv.Learner(data, model, callback_fns=SequentialInstructor.partial(instructions), **learn_kwargs)
    learn.split((learn.model.bu[3], learn.model.td[0]))
    if pretrained:
        learn.freeze()
    return learn


class SequentialInstructor(fv.LearnerCallback):
    def __init__(self, learn, instructions):
        super().__init__(learn)
        self.instructions = torch.tensor(instructions)
        self.reindex = self.instructions.argsort()

    @classmethod
    def partial(cls, instructions):
        def _inner(learn):
            return cls(learn, instructions)

        return _inner

    def on_batch_begin(self, last_input, **kwargs: Any):
        self.learn.model.clear()
        batch_size = last_input.shape[0]
        instructions = last_input.new(self.instructions, dtype=torch.long).expand(batch_size, len(self.instructions)).T
        return {'last_input': (last_input, instructions)}

    def on_loss_begin(self, last_output, **kwargs):
        bu_out, td_out = last_output
        return {'last_output': (bu_out[self.reindex], td_out[self.reindex])}
