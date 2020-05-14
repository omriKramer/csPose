from fastai.vision import *
import parts


def test_cs():
    tree = parts.ObjectTree.from_meta_folder('./broden_meta')
    body = create_body(models.resnet34, False)
    m = parts.CsNet(body, tree)
    img = torch.rand([8, 3, 256, 256])
    gt = torch.randint(low=0, high=4, size=(8, 256, 256)), []
    out = m(img, gt)
    assert out is not None
