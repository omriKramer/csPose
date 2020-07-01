import torch
from fastai.layers import embedding, conv2d
from torch import nn

import utils
from models import layers, Instructor
from models.upernet import get_fpn


class TaskMod(nn.Module):

    def __init__(self, instructor, fpn, dims, fpn_dim=512):
        super().__init__()
        self.fpn = fpn
        self.dims = dims
        self.embeddings = nn.ModuleList([embedding(instructor.tree.n_obj, d) for d in dims])
        self.classifier = nn.Sequential(layers.conv_layer(fpn_dim, fpn_dim), conv2d(fpn_dim, 1, ks=1, bias=True))
        self.instructor = instructor

    def forward(self, img):
        out = []
        for inst in self.instructor.inst:
            vecs = [e(inst) for e in self.embeddings]
            f = self.fpn(img, vecs[:-1])
            f = f * vecs[-1][:, :, None, None]
            out.append(self.classifier(f))

        out = torch.cat(out, dim=1)
        return out


def taskmod(root, tree, obj_classes=None):
    encoder_path, decoder_path = utils.upernet_ckpt(root)
    instructor = Instructor(tree, obj_classes=obj_classes)
    fpn = get_fpn(tree, weights_encoder=encoder_path, weights_decoder=decoder_path, task_modulation=True)
    encoder_channels = [128, 256, 512, 1024]
    decoder_cahnnels = [2048, 512, 512, 512]
    dims = encoder_channels + decoder_cahnnels + [512]
    model = TaskMod(instructor, fpn, dims)
    return model, instructor
