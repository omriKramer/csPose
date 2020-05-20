from fastai.vision import *
import parts


class StubLearn:
    pass


def test_cs():
    tree = parts.ObjectTree.from_meta_folder('./broden_meta')
    obj_gt = torch.randint(low=0, high=6, size=(8, 256, 256))
    part_gt = torch.randint(low=-1, high=3, size=(8, 256, 256))
    learn = StubLearn()
    metrics = parts.BrodenMetrics(learn, tree, preds_func=tree.cs_preds_func, restrict=False)
    part_gt[part_gt == 1] = 82
    part_gt[part_gt == 2] = 5
    gt = obj_gt, part_gt
    body = create_body(models.resnet18, False)
    m = parts.CsNet(body, tree)
    img = torch.rand([8, 3, 256, 256])
    out = m(img, gt)
    crit = parts.Loss(tree)
    loss = crit(out, obj_gt, part_gt)
    metrics.on_epoch_begin()
    metrics.on_batch_end(out, gt, False)
