import torch
from fastai.layers import embedding, conv2d
from torch import nn

import utils
from models import layers, Instructor
from models.upernet import get_fpn


class TaskMod(nn.Module):

    def __init__(self, instructor, fpn, dims, fpn_dim=512, full_head=False, obj_classes=None):
        super().__init__()
        self.fpn = fpn
        self.dims = dims
        self.embeddings = nn.ModuleList([embedding(instructor.tree.n_obj, d) for d in dims])
        c = len(obj_classes) if full_head else 1
        self.classifier = nn.Sequential(layers.conv_layer(fpn_dim, fpn_dim), conv2d(fpn_dim, c, ks=1, bias=True))
        self.instructor = instructor
        self.inst2idx = None
        if full_head:
            self.inst2idx = {inst: idx for idx, inst in enumerate(obj_classes)}

    def forward(self, img):
        out = []
        for inst in self.instructor.inst:
            vecs = [e(inst) for e in self.embeddings]
            f = self.fpn(img, vecs[:-1])
            f = f * vecs[-1][:, :, None, None]
            p = self.classifier(f)
            if self.inst2idx:
                idx = [self.inst2idx[i] for i in inst.cpu().tolist()]
                p = p[range(len(p)), idx]
                p = p[:, None]
            out.append(p)

        out = torch.cat(out, dim=1)
        return out


def taskmod(root, tree, full_head=False, obj_classes=None):
    encoder_path, decoder_path = utils.upernet_ckpt(root)
    instructor = Instructor(tree, obj_classes=obj_classes)
    fpn = get_fpn(tree, weights_encoder=encoder_path, weights_decoder=decoder_path, task_modulation=True)
    encoder_channels = [128, 256, 512, 1024]
    decoder_channels = [2048, 512, 512, 512]
    dims = encoder_channels + decoder_channels + [512]
    model = TaskMod(instructor, fpn, dims, full_head=full_head, obj_classes=obj_classes)
    return model, instructor
