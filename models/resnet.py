import itertools

import torch
from torch import nn
from torch.nn import functional as F

from models.counter_stream import CSBlock


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class InterpolateConv(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor, mode='bilinear', align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.conv = conv1x1(in_planes, out_planes)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        x = self.conv(x)
        return x


class CSConv(CSBlock):

    def __init__(self, in_planes, bu_planes, td_planes, kernel_size, stride=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.bu_conv = nn.Conv2d(in_planes, bu_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)
        self.bu_bn = nn.BatchNorm2d(bu_planes)
        self.bu_multp = conv1x1(bu_planes, bu_planes)
        self.bu_side_bn = nn.BatchNorm2d(bu_planes)

        self.td_conv = nn.Conv2d(bu_planes, td_planes, kernel_size=kernel_size, bias=bias, padding=padding)
        self.td_multp = conv1x1(bu_planes, bu_planes)
        self.td_side_bn = nn.BatchNorm2d(bu_planes)

        self.upsample = None
        if stride == 2:
            self.upsample = InterpolateConv(bu_planes, bu_planes, 2)
        elif stride > 2:
            raise ValueError('Supports stride of 1 or 2')

        self.relu = nn.ReLU(inplace=True)
        self.bu_out = self.td_in = None

    def clear(self):
        self.bu_out = self.td_in = None

    def _bottom_up(self, x):
        x = self.bu_conv(x)
        x = self.bu_bn(x)
        x = self.relu(x)

        if self.td_in is not None:
            x = self.bu_multp(self.td_in) + x
            x = self.bu_side_bn(x)
            x = self.relu(x)

        self.bu_out = x
        return x

    def _top_down(self, x):
        x = self.td_multp(self.bu_out) + x
        x = self.td_side_bn(x)
        x = self.relu(x)

        self.td_in = x
        if self.upsample:
            x = self.upsample(x)

        x = self.td_conv(x)
        return x


class BasicBlock(CSBlock):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, upsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.bu_multp1 = conv1x1(planes, planes)
        self.bu_side_bn1 = nn.BatchNorm2d(planes)
        self.bu_multp2 = conv1x1(planes, planes)
        self.bu_side_bn2 = nn.BatchNorm2d(planes)

        self.td_conv1 = conv3x3(planes, planes)
        self.td_bn1 = norm_layer(planes)
        self.td_conv2 = conv3x3(planes, inplanes)
        self.td_bn2 = norm_layer(inplanes)
        self.upsample = upsample

        self.td_multp1 = conv1x1(planes, planes)
        self.td_side_bn1 = nn.BatchNorm2d(planes)
        self.td_multp2 = conv1x1(planes, planes)
        self.td_side_bn2 = nn.BatchNorm2d(planes)

        self.bu_out1 = self.bu_out2 = None
        self.td_in1 = self.td_in2 = None

    def clear(self):
        self.bu_out1 = self.bu_out2 = None
        self.td_in1 = self.td_in2 = None

    def _bottom_up(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.td_in2 is not None:
            out = self.bu_multp1(self.td_in2) + out
            out = self.bu_side_bn1(out)
            out = self.relu(out)

        self.bu_out1 = out

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.td_in1 is not None:
            out = self.bu_multp2(self.td_in1) + out
            out = self.bu_side_bn2(out)
            out = self.relu(out)

        self.bu_out2 = out

        return out

    def _top_down(self, x):
        x = self.td_multp1(self.bu_out2) + x
        x = self.td_side_bn1(x)
        identity = self.relu(x)

        self.td_in1 = identity

        out = self.td_conv1(identity)
        out = self.td_bn1(out)
        out = self.relu(out)

        out = self.td_multp2(self.bu_out1) + out
        out = self.td_side_bn2(out)
        out = self.relu(out)

        self.td_in2 = out

        if self.stride == 2:
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

        out = self.td_conv2(out)
        out = self.td_bn2(out)
        if self.upsample:
            identity = self.upsample(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, layers_out=3, num_instructions=10, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv_block = CSConv(3, self.inplanes, layers_out, kernel_size=7, stride=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.embedding = nn.Embedding(num_instructions, 512)
        self.td_fc = nn.Linear(2 * 512, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            upsample = nn.Sequential(
                InterpolateConv(planes, self.inplanes * block.expansion, scale_factor=stride),
                norm_layer(self.inplanes)
            )

        layers = [
            block(self.inplanes, planes, stride=stride, downsample=downsample, upsample=upsample, groups=self.groups,
                  base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.ModuleList(layers)

    def forward(self, x, commands):
        td = []
        for cmd in commands:
            out_bu = x
            for l in self._iter_inner():
                out_bu = l(out_bu, 'BU')

            pre_pooling_size = out_bu.shape
            out_bu = self.avgpool(out_bu)
            out_bu = torch.flatten(out_bu, 1)

            bu_features = out_bu
            # out_bu = self.fc(out_bu)

            out_td = self.embedding(cmd)
            out_td = out_td.expand(bu_features.shape[0], -1)
            out_td = torch.cat((out_td, bu_features), dim=1)
            out_td = self.td_fc(out_td)
            out_td = self.relu(out_td)

            out_td = out_td[:, :, None, None].expand(pre_pooling_size)
            for l in reversed(list(self._iter_inner())):
                out_td = l(out_td, 'TD')
            td.append(out_td.squeeze())

        self.clear()
        results = {
            'td': torch.stack(td, dim=1)
        }
        return results

    def _iter_inner(self):
        iterator = itertools.chain([self.conv_block], self.layer1, self.layer2, self.layer3, self.layer4)
        return iterator

    def clear(self):
        for l in self._iter_inner():
            l.clear()


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)
