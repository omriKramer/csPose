import torch
from fastai.layers import embedding, conv2d
from fastai.vision import add_metrics
from torch import nn

import utils
from models import nnlayers, upernet, Instructor

encoder_channels = [128, 256, 512, 1024]
decoder_channels = [2048, 512, 512, 512]
mod_dims = encoder_channels + decoder_channels + [512]


def init_ones(m):
    if type(m) == nn.Embedding:
        nn.init.ones_(m.weight)


def iter_and_embed(x, layers, vecs, start=0):
    out = []
    for i, layer in enumerate(layers[start:], start=start):
        if vecs is not None:
            x = x * vecs[i][:, :, None, None]
        x = layer(x)
        out.append(x)
    return out


class ModModel(nn.Module):

    def __init__(self, instructor, fpn, classifier):
        super().__init__()
        self.fpn = fpn
        self.classifier = classifier
        self.embeddings = nn.ModuleList([embedding(instructor.tree.n_obj, d) for d in mod_dims])
        self.instructor = instructor

    def forward(self, img):
        out = []
        for inst in self.instructor.inst:
            vecs = [e(inst) for e in self.embeddings]
            f = self.fpn(img, vecs[:-1])
            f = f * vecs[-1][:, :, None, None]
            p = self.classifier(f)
            out.append(p)
        return out


class TaskMod(ModModel):

    def __init__(self, instructor, fpn, fpn_dim=512):
        classifier = nn.Sequential(nnlayers.conv_layer(fpn_dim, fpn_dim), conv2d(fpn_dim, 1, ks=1, bias=True))
        super().__init__(instructor, fpn, classifier)


class ModFPN(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img, vecs=None, **kwargs):
        encoder_vecs, decoder_vecs = None, None
        if vecs:
            encoder_vecs = vecs[:4]
            decoder_vecs = vecs[4:]
        out = self.encoder(img, vecs=encoder_vecs, return_feature_maps=True, **kwargs)
        out = self.decoder(out, vecs=decoder_vecs)
        return out


class ModulationEncoder(nn.Module):

    def __init__(self, head, layers):
        super().__init__()
        self.head = head
        self.layers = layers

    def forward(self, x, vecs=None, **kwargs):
        x = self.head(x)
        out = []
        for v, layer in zip(vecs, self.layers):
            x = x * v[:, :, None, None]
            x = layer(x)
            out.append(x)
        return out


def deconstruct_fpn(fpn: upernet.FPN):
    encoder = fpn.encoder
    decoder = fpn.decoder

    head = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu1,
                         encoder.conv2, encoder.bn2, encoder.relu2,
                         encoder.conv3, encoder.bn3, encoder.relu3,
                         encoder.maxpool)
    encoder_layers = nn.ModuleList([encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4])
    return head, encoder_layers, decoder


def add_taskmod(fpn: upernet.FPN):
    head, encoder_layers, decoder = deconstruct_fpn(fpn)
    encoder = ModulationEncoder(head, encoder_layers)
    model = ModFPN(encoder, decoder)
    return model


def taskmod(root, tree, obj_classes=None):
    encoder_path, decoder_path = utils.upernet_ckpt(root)
    instructor = Instructor(tree, obj_classes=obj_classes)
    fpn = upernet.get_fpn(tree, weights_encoder=encoder_path, weights_decoder=decoder_path)
    fpn = add_taskmod(fpn)
    model = TaskMod(instructor, fpn)
    return model, instructor


def upernetmod(root, tree, instructor):
    encoder_path, decoder_path = utils.upernet_ckpt(root)
    seg_model = upernet.get_upernet(tree, weights_encoder=encoder_path, weights_decoder=decoder_path)
    fpn = upernet.extract_fpn(seg_model)
    fpn = add_taskmod(fpn)
    model = ModModel(instructor, fpn, seg_model.decoder.object_head)
    return model


class FixUpAdditionEncoder(nn.Module):

    def __init__(self, head, layers):
        super().__init__()
        self.head = head
        self.layers = layers

    def forward(self, x, vecs=None, last_output=None, **kwargs):
        x = self.head(x)
        if vecs is not None:
            x = x * vecs[0][:, :, None, None]
        if last_output is not None:
            x = x + last_output
        x = self.layers[0](x)

        out = [x]
        for i, layer in enumerate(self.layers[1:], start=1):
            if vecs is not None:
                x = x * vecs[i][:, :, None, None]
            x = layer(x)
            out.append(x)
        return out


class FixUpCatEncoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x, vecs=None, last_output=None, last_output_down=None, **kwargs):
        if vecs is not None:
            x = x * vecs[0][:, :, None, None]
        x = self.layers[0][0](x, last_output=last_output, last_output_down=last_output_down)
        x = self.layers[0][1:](x)
        out = [x]
        out.extend(iter_and_embed(x, self.layers, vecs, start=1))
        return out


class FixUPAddModel(nn.Module):

    def __init__(self, fpn, classifier, n_obj):
        super().__init__()
        self.fpn = fpn
        self.classifier = classifier
        self.embeddings = nn.ModuleList([embedding(1, d) for d in mod_dims])
        self.output_conv = nn.Sequential(nnlayers.conv_layer(n_obj, 128), nnlayers.conv_layer(128, 128))
        self.register_buffer('inst', torch.zeros(1, dtype=torch.long))
        self.n_obj = n_obj

    def forward(self, img):
        n, _, h, w = img.shape
        out1 = self.classifier(self.fpn(img))

        vecs = [e(self.inst).expand(n, -1) for e in self.embeddings]
        f = self.fpn(img, vecs=vecs[:-1], last_output=self.output_conv(out1.detach()))
        f = f * vecs[-1][:, :, None, None]
        out2 = self.classifier(f)
        return [out1, out2]


class FixUpBottleneck(nn.Module):

    def __init__(self, bottleneck):
        super().__init__()
        self.conv1 = bottleneck.conv1
        self.bn1 = bottleneck.bn1
        self.relu = nn.ReLU(inplace=True)
        self.conv_layer2 = nn.Sequential(bottleneck.conv2, bottleneck.bn2, nn.ReLU(inplace=True))
        self.conv_layer3 = nn.Sequential(bottleneck.conv3, bottleneck.bn3)
        self.down_conv = bottleneck.downsample[0]
        self.down_bn = bottleneck.downsample[1]

    def forward(self, x, last_output=None, last_output_down=None):
        identity = x
        x = self.conv1(x)
        if last_output is not None:
            x = x + last_output
        x = self.relu(self.bn1(x))
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)

        identity = self.down_conv(identity)
        if last_output_down is not None:
            identity = identity + last_output_down
        identity = self.down_bn(identity)

        x = x + identity
        x = self.relu(x)
        return x


class FixUPCatModel(nn.Module):

    def __init__(self, head, fpn, classifier, n_obj, n_iter=1):
        super().__init__()
        self.head = head
        self.fpn = fpn
        self.classifier = classifier
        e_dims = [n_obj] + mod_dims
        self.embeddings = nn.ModuleList([embedding(n_iter, d) for d in e_dims])
        self.register_buffer('inst', torch.arange(n_iter))
        self.n_obj = n_obj
        self.output_conv = nn.Conv2d(n_obj, 64, kernel_size=1, bias=False)
        self.down_conv = nn.Conv2d(n_obj, 256, kernel_size=1, bias=False)
        self._init()

    @torch.no_grad()
    def _init(self):
        self.embeddings.apply(init_ones)
        self.embeddings[0].weight.zero_()

    def forward(self, img):
        img = self.head(img)
        last_output = self.classifier(self.fpn(img))

        out = [last_output]
        for inst in self.inst:
            vecs = [e(inst).expand(len(img), -1) for e in self.embeddings]
            last_output = last_output.detach() * vecs[0][:, :, None, None]
            last_output_f = self.output_conv(last_output)
            last_output_d = self.down_conv(last_output)
            f = self.fpn(img, vecs=vecs[1:-1], last_output=last_output_f, last_output_down=last_output_d)
            f = f * vecs[-1][:, :, None, None]
            last_output = self.classifier(f)
            out.append(last_output)
        return out


def add_conv_channels(conv, n):
    new_conv = nn.Conv2d(conv.in_channels + n, conv.out_channels, conv.kernel_size, stride=conv.stride, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :conv.in_channels].copy_(conv.weight)
    return new_conv


def addition_fixup(fpn):
    head, encoder_layers, decoder = deconstruct_fpn(fpn)
    encoder = FixUpAdditionEncoder(head, encoder_layers)
    new_fpn = ModFPN(encoder, decoder)
    return new_fpn


@torch.no_grad()
def enlarge_conv(conv: nn.Conv2d, c) -> nn.Conv2d:
    in_c = conv.in_channels + c
    out_c = conv.out_channels
    ks = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    new_conv = nn.Conv2d(in_c, out_c, ks, stride=stride, padding=padding, bias=False)
    nn.init.kaiming_normal_(new_conv.weight)
    new_conv.weight[:, :conv.in_channels].copy_(conv.weight)
    return new_conv


def cat_fixup(fpn):
    head, encoder_layers, decoder = deconstruct_fpn(fpn)
    encoder_layers[0][0] = FixUpBottleneck(encoder_layers[0][0])
    encoder = FixUpCatEncoder(encoder_layers)
    new_fpn = ModFPN(encoder, decoder)
    return head, new_fpn


def fixup_upernet(root, tree, fix_type='add', n_iter=1):
    encoder_path, decoder_path = utils.upernet_ckpt(root)
    seg_model = upernet.get_upernet(tree, weights_encoder=encoder_path, weights_decoder=decoder_path)
    fpn = upernet.extract_fpn(seg_model)
    if fix_type == 'add':
        fpn = addition_fixup(fpn)
        model = FixUPAddModel(fpn, seg_model.decoder.object_head, tree.n_obj)
        return model
    elif fix_type == 'cat':
        head, fpn = cat_fixup(fpn)
        model = FixUPCatModel(head, fpn, seg_model.decoder.object_head, tree.n_obj, n_iter=n_iter)
        return model

    raise ValueError


class SingleObjLoss(nn.Module):

    def __init__(self, instructor, obj_class, pos_weight=None):
        super().__init__()
        self.instructor = instructor
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.obj_class = obj_class

    def forward(self, pred, obj_gt, part_gt):
        if not self.instructor.train:
            return torch.zeros(1)

        pred = [p.squeeze(dim=1) for p in pred]
        obj_gt = obj_gt.squeeze(dim=1)
        pred_size = pred[0].shape[-2:]
        obj_gt = utils.resize(obj_gt, pred_size)

        if len(pred) == 1:
            return self.binary_loss(pred[0], obj_gt, self.instructor.inst[0])
        elif len(pred) == 2:
            ce = self.ce(pred[0], obj_gt)
            bce = self.binary_loss(pred[1], obj_gt, self.instructor.inst[0])
            return ce + bce

        raise ValueError()

    def binary_loss(self, pred, gt, inst):
        assert torch.all(inst == self.obj_class)

        obj_mask = gt == inst[:, None, None]
        obj_target = obj_mask.float()
        pred = utils.binary_score(pred, self.obj_class)
        loss = self.bce(pred.view(-1, 1), obj_target.view(-1, 1))
        return loss


class TaskModPred:

    def __init__(self, obj):
        self.obj = obj

    def __call__(self, last_output, gt_size):
        obj_pred = utils.resize(last_output[0], gt_size)
        c_pred = utils.resize(last_output[1], gt_size)

        top2 = torch.topk(obj_pred, 2, dim=1)[1]
        b_pred = utils.binary_score(c_pred, self.obj)
        # sigmoid(0.5) = 0
        is_obj = b_pred > 0
        c = torch.tensor(self.obj, dtype=torch.long, device=top2.device)
        pred = torch.where(is_obj, c, top2[:, 0])
        pred = pred.where(is_obj + (pred != self.obj), top2[:, 1])
        return pred, None


class TaskmodMetrics(utils.LearnerMetrics):

    def __init__(self, learn, n_classes, pred_func, obj, target_func=None):
        super().__init__(learn, ['P.A.', 'mIoU', 'FP/FN'])
        self.target_func = target_func if target_func else utils.cm_target_func
        self.pred_func = pred_func
        self.cm = utils.ConfusionMatrix(n_classes)
        self.obj = obj

    def _reset(self):
        self.cm.reset()

    def on_batch_end(self, last_output, last_target, train, **kwargs):
        if train:
            return

        pred = self.pred_func(last_output, last_target)
        target = self.target_func(last_target)
        self.cm.update(pred, target)

    def on_epoch_end(self, last_metrics, **kwargs):
        matrix = self.cm.matrix
        objs = list(range(1, len(matrix)))
        pa = (matrix[objs, objs].sum() / matrix[1:].sum())
        gt_count = matrix[1:].sum(axis=1)
        pred_count = matrix[1:, 1:].sum(axis=0)
        intersection = matrix[objs, objs]
        iou = intersection / (gt_count + pred_count - intersection)
        miou = iou.mean()

        fp = matrix[1:, self.obj].sum() - matrix[self.obj, self.obj]
        fn = matrix[self.obj, :].sum() - matrix[self.obj, self.obj]
        return add_metrics(last_metrics, [pa, miou, fp/fn])
