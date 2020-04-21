import itertools

import fastai.vision as fv
import pytest
import torch
import torchvision
from torch.utils.data import TensorDataset

import models.cs_v2 as cs


@pytest.fixture(scope="module", params=[fv.models.resnet18, fv.models.resnet34])
def arch(request):
    return request.param


@pytest.fixture()
def bu(arch):
    return fv.create_body(arch, False)


@pytest.fixture(scope="module", params=[0, 16])
def bu_c(request):
    return request.param


def rand_target(n):
    t = torch.rand(n, 16, 3)
    t[..., 2].round_()
    return t


@pytest.fixture(scope="module")
def databunch():
    shape = 3, 128, 128

    train = TensorDataset(torch.rand(8, *shape), rand_target(8))
    val = TensorDataset(torch.rand(4, *shape), rand_target(4))
    train.c = 16
    db = fv.ImageDataBunch.create(train, val, bs=4, num_workers=0)
    return db


@pytest.fixture(scope="module")
def learn(databunch, arch, bu_c, single_instructor):
    return cs.cs_learner(databunch, arch, single_instructor, td_c=16, bu_c=bu_c)


def test_counter_stream_init(bu):
    nk = 16
    img_size = 128, 128
    instructor = cs.RecurrentInstructor(1)
    cs_net = cs.CounterStream(bu, instructor, td_c=nk, embedding=None, img_size=img_size, ppm=True, fuse=True)
    cs_net.eval()
    img = torch.rand(1, 3, *img_size)
    td_out = cs_net(img)
    td_out_size = 1, nk, img_size[0] / 4, img_size[1] / 4
    assert td_out.shape == td_out_size


def test_cs_learner_freeze(learn):
    model = learn.model
    for layer in fv.flatten_model(model.bu_body):
        should_require = False
        if isinstance(layer, fv.bn_types):
            should_require = True
        for p in layer.parameters():
            assert p.requires_grad == should_require

    branches = [model.td, model.laterals, model.emb]
    if model.bu_head:
        branches.append(model.bu_head)
    for p in itertools.chain.from_iterable(b.parameters() for b in branches):
        assert p.requires_grad


def first_conv(module):
    for l in module:
        if isinstance(l, torch.nn.Conv2d):
            return l
        if isinstance(l, (torchvision.models.resnet.BasicBlock, cs.TDBlock)):
            return l.conv1


def assert_params(m, params):
    for (name, p1), p2 in zip(m.named_parameters(), params):
        assert (p1 == p2).all()


class HookOutputs:
    def __init__(self, model):
        self.hooks = []
        for l in model:
            self.hooks.append(l.register_forward_hook(self.hk_fn))
        self.outputs, self.inputs, self.params = [], [], []

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.outputs = []
        self.inputs = []
        self.params = []

    def hk_fn(self, module, inp, output):
        self.outputs.append(output.detach())
        self.inputs.append(inp[0].detach())
        self.params.append([p.detach().clone() for p in module.parameters()])


class HookAndAssert:
    def __init__(self, model, expected_out, expected_in, expected_params):
        self.expected_params = expected_params
        self.expected_in = expected_in
        self.expected_out = expected_out
        self.i = 0
        self.hooks = []
        for l in model:
            self.hooks.append(l.register_forward_hook(self.hk_fn))

    def hk_fn(self, module, inp, output):
        for p in module.parameters():
            assert not p.requires_grad

        if self.i < len(self.expected_in):
            assert (inp[0] == self.expected_in[self.i]).all()
            assert_params(module, self.expected_params[self.i])
            assert (output == self.expected_out[self.i]).all()
            self.i += 1


def disable_grads(layer):
    for p in layer.parameters():
        p.requires_grad = False


def freeze_except_td_laterals(model):
    for l in (model.bu_body, model.td):
        disable_grads(l)

    disable_grads(model.laterals[:17])


def test_double_unet(bu):
    nk = 16
    img_size = 128, 128
    dunet = cs.DoubleUnet(bu)
    out = dunet(torch.rand(2, 3, *img_size))
    assert len(out) == 2
