from fastai.vision import *

import parts
import utils
from models import layers
from models.upernet import get_fpn
from parts import upernet_data_pipeline


class Loss:

    def __init__(self, tree):
        self.tree = tree
        self.obj_ce = nn.CrossEntropyLoss(ignore_index=0)
        self.part_ce = nn.CrossEntropyLoss(ignore_index=-1)

    def __call__(self, pred, obj_gt, part_gt):
        obj_pred, part_pred = pred
        obj_gt = obj_gt.squeeze(dim=1)
        part_gt = part_gt.squeeze(dim=1)

        pred_size = obj_pred.shape[-2:]
        obj_gt = utils.resize(obj_gt, pred_size)
        part_gt = utils.resize(part_gt, pred_size)
        part_gt = self.tree.split_parts_gt(obj_gt, part_gt, mark_in_obj=False)

        obj_loss = self.obj_ce(obj_pred, obj_gt)
        part_loss = []
        for o, (start, end) in self.tree.obj2part_idx.items():
            obj_mask = obj_gt == o
            o_part_gt = part_gt[self.tree.obj2idx[o]].clone()
            o_part_gt[~obj_mask] = -1
            o_part_pred = part_pred[:, start:end]
            part_loss.append(self.part_ce(o_part_pred, o_part_gt))

        loss = obj_loss + sum(part_loss)
        return loss


class Head3(nn.Module):

    def __init__(self, tree, weights_encoder='', weights_decoder=''):
        super().__init__()
        self.fpn = get_fpn(tree, weights_encoder=weights_encoder, weights_decoder=weights_decoder)
        fpn_dim = 512
        self.obj_branch = nn.ModuleList([layers.conv_layer(fpn_dim, fpn_dim),
                                         conv2d(fpn_dim, tree.n_obj, ks=1, bias=True)])
        self.bu = nn.ModuleList([layers.conv_layer(tree.n_obj, fpn_dim // 2), layers.conv_layer(fpn_dim, fpn_dim)])
        self.part_branch = nn.Sequential(layers.conv_layer(fpn_dim, fpn_dim),
                                         conv2d(fpn_dim, tree.n_parts, ks=1, bias=True))
        self.lateral = layers.conv_layer(fpn_dim, fpn_dim // 2)

    def forward(self, img):
        features = self.fpn(img)
        obj_hidden = self.obj_branch[0](features)
        obj_pred = self.obj_branch[1](obj_hidden)
        x = self.bu[0](obj_pred.detach())
        x = torch.cat([x, self.lateral(obj_hidden)], dim=1)
        x = self.bu[1](x)
        x = features + x
        part_pred = self.part_branch(x)
        return obj_pred, part_pred


def main(args):
    broden_root = Path(args.root).resolve()
    tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
    encoder_path, decoder_path = utils.upernet_ckpt(broden_root)
    model = Head3(tree, encoder_path, decoder_path)
    db = upernet_data_pipeline(broden_root)

    metrics = partial(parts.BrodenMetricsClbk, obj_tree=tree, split_func=lambda o: o)

    learn = Learner(db, model, loss_func=Loss(tree), callback_fns=metrics, )
    learn.split((learn.model.obj_branch,))
    learn.freeze()

    utils.fit_and_log(learn, 'object-P.A', save=args.save, epochs=20, lr=1e-2, pct_start=args.pct_start)


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    parser.add_argument('--root', default='unifiedparsing/broden_dataset')
    main(parser.parse_args())
