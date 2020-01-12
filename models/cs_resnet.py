import itertools
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from models.counter_stream import CSBlock


def conv_transpose3x3(in_planes, out_planes, stride=1):
    """3x3 transposed convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=stride - 1,
                              bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def disable_grads(*layers):
    for l in layers:
        for p in l.parameters():
            p.requires_grad = False


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
        self.bu_conv = nn.Conv2d(in_planes, bu_planes, kernel_size, stride, padding, bias=bias)
        self.bu_bn = nn.BatchNorm2d(bu_planes)
        self.bu_lateral = conv1x1(bu_planes, bu_planes)

        self.td_conv = nn.ConvTranspose2d(bu_planes, td_planes, kernel_size, stride, padding, output_padding=1,
                                          bias=False)
        self.td_lateral = conv1x1(bu_planes, bu_planes)

        self.relu = nn.ReLU(inplace=True)
        self.bu_out = self.td_in = None

    def clear(self):
        self.bu_out = self.td_in = None

    def _bottom_up(self, x):
        x = self.bu_conv(x)
        x = self.bu_bn(x)
        x = self.relu(x)

        if self.td_in is not None:
            x = x + self.bu_lateral(self.td_in)

        self.bu_out = x.detach()
        return x

    def _top_down(self, x):
        x = x + self.td_lateral(self.bu_out)
        self.td_in = x.detach()

        x = self.td_conv(x)
        return x

    def one_iteration(self):
        disable_grads(self.bu_lateral)


class BasicBlock(CSBlock):
    expansion = 1
    depth = 2

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

        self.td_conv1 = conv3x3(planes, planes)
        self.td_bn1 = norm_layer(planes)
        self.td_conv2 = conv_transpose3x3(planes, inplanes, stride)
        self.td_bn2 = norm_layer(inplanes)
        self.upsample = upsample

        self.bu_lateral = conv1x1(planes, planes)
        self.td_lateral = conv1x1(planes, planes)

        self.bu_out = None
        self.td_in = None

    def clear(self):
        self.bu_out = None
        self.td_in = None

    def one_iteration(self):
        disable_grads(self.bu_lateral, self.td_lateral)

    def _bottom_up(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.td_in is not None:
            out = self.bu_lateral(self.td_in) + out

        self.bu_out = out.detach()
        return out

    def _top_down(self, x):
        x = self.td_lateral(self.bu_out) + x
        identity = x
        self.td_in = identity.detach()

        out = self.td_conv1(x)
        out = self.td_bn1(out)
        out = self.relu(out)

        out = self.td_conv2(out)
        out = self.td_bn2(out)

        if self.upsample:
            identity = self.upsample(identity)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(CSBlock):
    expansion = 4
    depth = 3

    def __init__(self, inplanes, planes, stride=1, downsample=None, upsample=None,
                 base_width=64, norm_layer=None):
        assert (downsample and upsample) or (not downsample and not upsample)

        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample

        self.upsample = upsample
        self.td_conv1 = conv1x1(planes * self.expansion, width)
        self.td_bn1 = norm_layer(width)
        self.td_conv2 = conv_transpose3x3(width, width, stride)
        self.td_bn2 = norm_layer(width)
        self.td_conv3 = conv1x1(width, inplanes)
        self.td_bn3 = norm_layer(inplanes)

        self.td_in = self.bu_out = None
        self.bu_lateral = conv1x1(planes * self.expansion, planes * self.expansion)
        self.td_lateral = conv1x1(planes * self.expansion, planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def clear(self):
        self.td_in = self.bu_out = None

    def one_iteration(self):
        disable_grads(self.bu_lateral)

    def _bottom_up(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        if self.td_in is not None:
            out = self.bu_lateral(self.td_in) + out

        self.bu_out = out.detach()
        return out

    def _top_down(self, x):
        x = x + self.td_lateral(self.bu_out)
        identity = x
        self.td_in = x.detach()

        out = self.td_conv1(x)
        out = self.td_bn1(out)
        out = self.relu(out)

        out = self.td_conv2(out)
        out = self.td_bn2(out)
        out = self.relu(out)

        out = self.td_conv3(out)
        out = self.td_bn3(out)

        if self.upsample is not None:
            identity = self.upsample(identity)

        out = out + identity
        out = self.relu(out)

        return out


class CsResNet(CSBlock):

    def __init__(self, block, layers, td_outplanes=3, num_instructions=10, norm_layer=None):
        super(CsResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.base_width = 64
        self.conv_block = CSConv(3, self.inplanes, td_outplanes, kernel_size=7, stride=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.td_fc1 = nn.Linear(num_instructions, 512)
        self.td_fc2 = nn.Linear(512 + (512 * block.expansion), 512 * block.expansion)

        self.pre_pooling_shape = None
        self.bu_features = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        depth = sum(layers) * block.depth + 2
        self.name = self.__class__.__name__ + str(depth)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            upsample = nn.Sequential(
                InterpolateConv(planes * block.expansion, self.inplanes, scale_factor=stride),
                norm_layer(self.inplanes)
            )

        layers = [
            block(self.inplanes, planes, stride=stride, downsample=downsample, upsample=upsample,
                  base_width=self.base_width, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width, norm_layer=norm_layer))

        return nn.ModuleList(layers)

    def _bottom_up(self, x):
        for layer in self._iter_inner():
            x = layer(x, 'BU')

        self.pre_pooling_shape = x.shape

        out = self.avgpool(x)
        out = out.flatten(1)
        self.bu_features = out
        return None

    def _top_down(self, x):
        out = self.td_fc1(x)
        out = torch.cat((out, self.bu_features), dim=1)

        out = self.td_fc2(out)
        out = self.relu(out)

        out = out[:, :, None, None].expand(self.pre_pooling_shape)
        for layer in reversed(list(self._iter_inner())):
            out = layer(out, 'TD')

        out = out.squeeze(dim=1)
        return out

    def _iter_inner(self) -> Iterator[CSBlock]:
        iterator = itertools.chain([self.conv_block], self.layer1, self.layer2, self.layer3, self.layer4)
        return iterator

    def clear(self):
        self.pre_pooling_shape = self.bu_features = None
        for layer in self._iter_inner():
            layer.clear()

    def one_iteration(self):
        for layer in self._iter_inner():
            layer.one_iteration()


def resnet18(**kwargs):
    return CsResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return CsResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return CsResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
