import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet, resnext
from lib.nn import SynchronizedBatchNorm2d, PrRoIPool2D


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    @staticmethod
    def pixel_acc(pred, label, ignore_index=-1):
        _, preds = torch.max(pred, dim=1)
        valid = (label != ignore_index).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    @staticmethod
    def part_pixel_acc(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        mask_object = (gt_seg_object == object_label)
        _, pred = torch.max(pred_part, dim=1)
        acc_sum = mask_object * (pred == gt_seg_part)
        acc_sum = torch.sum(acc_sum.view(acc_sum.size(0), -1), dim=1)
        acc_sum = torch.sum(acc_sum * valid)
        pixel_sum = torch.sum(mask_object.view(mask_object.size(0), -1), dim=1)
        pixel_sum = torch.sum(pixel_sum * valid)
        return acc_sum, pixel_sum

    @staticmethod
    def part_loss(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        mask_object = (gt_seg_object == object_label)
        loss = F.nll_loss(pred_part, gt_seg_part * mask_object.long(), reduction='none')
        loss = loss * mask_object.float()
        loss = torch.sum(loss.view(loss.size(0), -1), dim=1)
        nr_pixel = torch.sum(mask_object.view(mask_object.shape[0], -1), dim=1)
        sum_pixel = (nr_pixel * valid).sum()
        loss = (loss * valid.float()).sum() / torch.clamp(sum_pixel, 1).float()
        return loss


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, tree, loss_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit_dict = nn.ModuleDict()
        if loss_scale is None:
            self.loss_scale = {"object": 1, "part": 0.5, "scene": 0.25, "material": 1}
        else:
            self.loss_scale = loss_scale

        # criterion
        self.crit_dict["object"] = nn.NLLLoss(ignore_index=0)  # ignore background 0
        self.crit_dict["material"] = nn.NLLLoss(ignore_index=0)  # ignore background 0
        self.crit_dict["scene"] = nn.NLLLoss(ignore_index=-1)  # ignore unlabelled -1
        self.tree = tree

    def forward(self, img):
        output_switch = {"object": True, "part": True, "scene": False, "material": False}
        pred = self.decoder(
            self.encoder(img, return_feature_maps=True),
            output_switch=output_switch
        )
        return pred

    def loss_func(self, pred, obj_gt, part_gt, valid_part):
        # loss
        loss_dict = {}
        if pred['object'] is not None:  # object
            loss_dict['object'] = self.crit_dict['object'](pred['object'], obj_gt)
        if pred['part'] is not None:  # part
            part_loss = 0
            for idx_part, object_label in enumerate(self.tree.obj_with_parts):
                # part_gt shape should be:(bs, n_obj_with_part, h, w)
                part_loss += self.part_loss(
                    pred['part'][idx_part], part_gt,
                    obj_gt, object_label, valid_part[:, idx_part])
            loss_dict['part'] = part_loss
        loss_dict['total'] = sum([loss_dict[k] * self.loss_scale[k] for k in loss_dict.keys()])

        # metric
        metric_dict = {}
        if pred['object'] is not None:
            metric_dict['object'] = self.pixel_acc(
                pred['object'], obj_gt, ignore_index=0)
        if pred['part'] is not None:
            acc_sum, pixel_sum = 0, 0
            for idx_part, object_label in enumerate(self.tree.obj_with_parts):
                acc, pixel = self.part_pixel_acc(
                    pred['part'][idx_part], part_gt, obj_gt,
                    object_label, valid_part[:, idx_part])
                acc_sum += acc
                pixel_sum += pixel
            metric_dict['part'] = acc_sum.float() / (pixel_sum.float() + 1e-10)

        return {'metric': metric_dict, 'loss': loss_dict}


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        SynchronizedBatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class ModelBuilder:
    def __init__(self):
        pass

    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        # elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='resnet50_dilated8', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext)  # we can still use class Resnet
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    def build_decoder(self, tree,
                      arch='ppm_bilinear_deepsup', fc_dim=512,
                      weights='', use_softmax=False):
        nr_classes = {'scene': 365, 'object': tree.n_obj, 'part': tree.n_parts, 'material': 26, 'texture': 47}
        if arch == 'upernet_lite':
            net_decoder = UPerNet(
                nr_classes,
                tree,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                nr_classes,
                tree,
                fc_dim=fc_dim,
                use_softmax=True,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


# upernet
class UPerNet(nn.Module):
    def __init__(self, nr_classes, tree, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        self.tree = tree
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            # we use the feature map size instead of input image size, so down_scale = 1.0
            self.ppm_pooling.append(PrRoIPool2D(scale, scale, 1.))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_fusion = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)

        # background included. if ignore in loss, output channel 0 will not be trained.
        self.nr_scene_class, self.nr_object_class, self.nr_part_class, self.nr_material_class = \
            nr_classes['scene'], nr_classes['object'], nr_classes['part'], nr_classes['material']

        # input: PPM out, input_dim: fpn_dim
        self.scene_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fpn_dim, self.nr_scene_class, kernel_size=1, bias=True)
        )

        # input: Fusion out, input_dim: fpn_dim
        self.object_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, self.nr_object_class, kernel_size=1, bias=True)
        )

        # input: Fusion out, input_dim: fpn_dim
        self.part_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, self.nr_part_class, kernel_size=1, bias=True)
        )

        # input: FPN_2 (P2), input_dim: fpn_dim
        self.material_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, self.nr_material_class, kernel_size=1, bias=True)
        )

    def forward(self, conv_out, output_switch=None):

        output_dict = {k: None for k in output_switch.keys()}

        conv5 = conv_out[-1]
        input_size = conv5.size()
        roi = []  # fake rois, just used for pooling
        for i in range(input_size[0]):  # batch size
            roi.append(torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1))  # b, x0, y0, x1, y1
        roi = torch.cat(roi, dim=0).type_as(conv5)
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(
                pool_scale(conv5, roi.detach()),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        if output_switch['scene']:  # scene
            output_dict['scene'] = self.scene_head(f)

        if output_switch['object'] or output_switch['part'] or output_switch['material']:
            fpn_feature_list = [f]
            for i in reversed(range(len(conv_out) - 1)):
                conv_x = conv_out[i]
                conv_x = self.fpn_in[i](conv_x)  # lateral branch

                f = F.interpolate(
                    f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)  # top-down branch
                f = conv_x + f

                fpn_feature_list.append(self.fpn_out[i](f))
            fpn_feature_list.reverse()  # [P2 - P5]

            # material
            if output_switch['material']:
                output_dict['material'] = self.material_head(fpn_feature_list[0])

            if output_switch['object'] or output_switch['part']:
                output_size = fpn_feature_list[0].size()[2:]
                fusion_list = [fpn_feature_list[0]]
                for i in range(1, len(fpn_feature_list)):
                    fusion_list.append(F.interpolate(
                        fpn_feature_list[i],
                        output_size,
                        mode='bilinear', align_corners=False))
                fusion_out = torch.cat(fusion_list, 1)
                x = self.conv_fusion(fusion_out)

                if output_switch['object']:  # object
                    output_dict['object'] = self.object_head(x)
                if output_switch['part']:
                    output_dict['part'] = self.part_head(x)

        return output_dict


def get_upernet(tree, weights_encoder='', weights_decoder='', replace_bn=True):
    fc_dim = 2048
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch='resnet50',
        fc_dim=fc_dim,
        weights=weights_encoder)
    net_decoder = builder.build_decoder(
        tree,
        arch='upernet',
        fc_dim=fc_dim,
        weights=weights_decoder,
        use_softmax=True)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, tree)
    if replace_bn:
        segmentation_module.apply(replace_sync_bn)
    return segmentation_module


def replace_sync_bn(m):
    replace = {name: regular_bn(module) for name, module in m.named_children()
               if isinstance(module, SynchronizedBatchNorm2d)}
    for name, bn in replace.items():
        setattr(m, name, bn)


def regular_bn(sync_bn):
    bn = nn.BatchNorm2d(sync_bn.num_features, eps=sync_bn.eps, momentum=sync_bn.momentum)
    state = sync_bn.state_dict()
    state = {key: state[key] for key in ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked')}
    bn.load_state_dict(state)
    return bn


class PPM(nn.Module):

    def __init__(self, ppm_pooling, ppm_conv, ppm_last_conv):
        super().__init__()
        self.ppm_last_conv = ppm_last_conv
        self.ppm_conv = ppm_conv
        self.ppm_pooling = ppm_pooling

    def forward(self, x):
        input_size = x.size()
        roi = []  # fake rois, just used for pooling
        for i in range(input_size[0]):  # batch size
            roi.append(torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1))  # b, x0, y0, x1, y1
        roi = torch.cat(roi, dim=0).type_as(x)
        out = [x]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            out.append(pool_conv(F.interpolate(
                pool_scale(x, roi.detach()),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        out = torch.cat(out, 1)
        out = self.ppm_last_conv(out)
        return out


class FpnTD(nn.Module):

    def __init__(self, ppm, fpn_in, fpn_out, conv_fusion):
        super().__init__()
        self.conv_fusion = conv_fusion
        self.fpn_out = fpn_out
        self.fpn_in = fpn_in
        self.ppm = ppm

    def forward(self, conv_out, vecs=None):
        conv5 = conv_out[-1]
        if vecs:
            conv5 = conv5 * vecs[0]
        f = self.ppm(conv5)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = F.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)  # top-down branch
            f = conv_x + f
            if vecs:
                f = f * vecs[len(conv_out) - 1 - i]

            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()  # [P2 - P5]

        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_fusion(fusion_out)
        return x


class FPN(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img, vecs=None):
        encoder_vecs, decoder_vecs = None, None
        if vecs:
            encoder_vecs = vecs[:4]
            decoder_vecs = vecs[4:]
        out = self.encoder(img, return_feature_maps=True, vecs=encoder_vecs)
        out = self.decoder(out, vecs=decoder_vecs)
        return out


def extract_fpn(seg_model: SegmentationModule, task_modulation=False):
    encoder = seg_model.encoder
    if task_modulation:
        head = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu1,
                             encoder.conv2, encoder.bn2, encoder.relu2,
                             encoder.conv3, encoder.bn3, encoder.relu3,
                             encoder.maxpool)
        layers = nn.ModuleList([encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4])
        encoder = ModulationEncoder(head, layers)

    decoder = seg_model.decoder
    ppm = PPM(decoder.ppm_pooling, decoder.ppm_conv, decoder.ppm_last_conv)
    decoder = FpnTD(ppm, decoder.fpn_in, decoder.fpn_out, decoder.conv_fusion)
    fpn = FPN(encoder, decoder)
    return fpn


def get_fpn(tree, weights_encoder='', weights_decoder='', task_modulation=False):
    seg_model = get_upernet(tree, weights_encoder=weights_encoder, weights_decoder=weights_decoder)
    fpn = extract_fpn(seg_model, task_modulation=task_modulation)
    return fpn


class ModulationEncoder(nn.Module):

    def __init__(self, head, layers):
        super().__init__()
        self.head = head
        self.layers = layers

    def forward(self, x, vecs=None):
        x = self.head(x)
        out = []
        for layer, v in zip(vecs, self.layers):
            x = x * v[:, :, None, None]
            x = layer(x)
            out.append(x)
        return out
