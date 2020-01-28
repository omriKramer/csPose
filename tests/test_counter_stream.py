import itertools

import fastai.vision as fv
import pytest
import torch
from torch.utils.data import TensorDataset

import models.cs_v2 as cs


@pytest.fixture(scope="module", params=[fv.models.resnet18, fv.models.resnet50])
def arch(request):
    return request.param


@pytest.fixture()
def bu(arch):
    return fv.create_body(arch, False)


@pytest.fixture(scope="module", params=[0, 16])
def bu_c(request):
    return request.param


@pytest.fixture(scope="module")
def databunch():
    shape = 3, 128, 128
    train = TensorDataset(torch.rand(8, *shape), torch.rand(8, 16, 2))
    val = TensorDataset(torch.rand(4, *shape), torch.rand(4, 16, 2))
    train.c = 16
    db = fv.ImageDataBunch.create(train, val, bs=4, num_workers=0)
    return db


@pytest.fixture(scope="module")
def learn(databunch, arch, bu_c):
    instructor = cs.SingleInstruction()
    return cs.cs_learner(databunch, arch, 16, instructor, bu_c=bu_c)


def test_counter_stream_init(bu):
    n_inst = 3
    img_size = 128, 128
    cs_net = cs.CounterStream(bu, 16, img_size=img_size)
    img = torch.rand(1, n_inst, *img_size)
    instructions = torch.arange(3)[:, None]
    bu_out, td_out = cs_net(img, instructions)
    td_out_size = 1, n_inst, img_size[0] / 4, img_size[1] / 4
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
