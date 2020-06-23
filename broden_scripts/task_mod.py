from fastai.vision import *

import parts
import utils
from models import Instructor, layers
from models.upernet import get_fpn
from parts import upernet_data_pipeline


class TaskMod(nn.Module):

    def __init__(self, instructor, fpn, dims, fpn_dim=512):
        super().__init__()
        self.fpn = fpn
        self.dims = dims
        self.embeddings = nn.ModuleList([embedding(instructor.n_inst, d) for d in dims])
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


def get_model(root, tree):
    encoder_path, decoder_path = utils.upernet_ckpt(root)
    instructor = Instructor(tree)
    fpn = get_fpn(tree, weights_encoder=encoder_path, weights_decoder=decoder_path, task_modulation=True)
    encoder_channels = [128, 256, 512, 1024]
    decoder_cahnnels = [2048, 512, 512, 512]
    dims = encoder_channels + decoder_cahnnels + [512]
    model = TaskMod(instructor, fpn, dims)
    return model, instructor


def main(args):
    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    db = upernet_data_pipeline(broden_root)
    model, instructor = get_model(broden_root, tree)
    metrics = partial(parts.BinaryBrodenMetrics, obj_tree=tree, thresh=0.5)
    learn = Learner(db, model, loss_func=instructor.loss, callbacks=[instructor], callback_fns=metrics)
    learn.split((learn.model.embeddings,))
    learn.freeze()

    utils.fit_and_log(learn, 'object-P.A', save=args.save, epochs=20, lr=1e-3, pct_start=args.pct_start)


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    main(parser.parse_args())
