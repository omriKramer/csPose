from abc import ABC
import itertools

import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class InterpolateConv(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor, mode='bilinear'):
        super().__init__()
        self.conv = conv1x1(in_planes, out_planes)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class CSModule(nn.Module, ABC):

    def forward(self, x, mode):
        if mode == 'BU':
            return self.bottom_up(x)
        elif mode == 'TD':
            return self.top_down(x)

    def bottom_up(self, x):
        raise NotImplementedError

    def top_down(self, x):
        raise NotImplementedError

    def clear(self):
        """Clear the inner state created by Bottom-up/Top-Down runs."""
        raise NotImplementedError


class CounterStreamBlock(CSModule):

    def __init__(self, in_planes, planes, stride=1, downsample=None, upsample=None):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.bu_multp1 = conv1x1(planes, planes)
        self.bu_side_bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.bu_multp2 = conv1x1(planes, planes)
        self.bu_side_bn2 = nn.BatchNorm2d(planes)

        self.bu_out1 = self.bu_out2 = None
        self.td_in1 = self.td_in2 = None

        self.td_multp1 = conv1x1(planes, planes)
        self.td_side_bn1 = nn.BatchNorm2d(planes)

        self.td_conv1 = conv3x3(planes, planes)
        self.td_bn1 = nn.BatchNorm2d(planes)

        self.td_multp2 = conv1x1(planes, planes)
        self.td_side_bn2 = nn.BatchNorm2d(planes)

        self.td_conv2 = conv3x3(planes, in_planes)
        self.td_bn2 = nn.BatchNorm2d(in_planes)

        self.downsample = downsample
        self.upsample = upsample
        self.stride = stride

    def bottom_up(self, x):
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

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        if self.td_in1 is not None:
            out = self.self.bu_multp2(self.td_in1) + out
            out = self.bu_side_bn2(out)
            out = self.relu(out)

        self.bu_out2 = out

        return out

    def top_down(self, x):
        x = self.td_multp1(self.bu_out2) + x
        x = self.td_side_bn1(x)
        identity = self.relu(x)
        self.td_in1 = identity

        out = self.td_conv1(identity)
        out = self.td_bn1(out)
        out = self.relu(out)

        out = self.bu_multp2(self.bu_out1) + out
        out = self.td_side_bn2(out)
        out = self.relu(out)
        self.td_in2 = out

        if self.stride == 2:
            out = F.interpolate(out, scale_factor=2, mode='bilinear')

        out = self.td_conv2(out)
        out = self.td_bn2(out)

        if self.upsample:
            identity = self.upsample(identity)

        out += identity
        out = self.relu(out)

        return out

    def clear(self):
        self.bu_out1 = self.bu_out2 = None
        self.td_in1 = self.td_in2 = None


class CounterStreamNet(CSModule):

    def __init__(self, layers, num_classes=10, num_instructions=10):
        super().__init__()
        planes = [int(64 * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(1, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu = nn.ReLU(inplace=True)

        self.bu_multp = nn.Parameter(torch.rand(planes[0]))
        self.bu_side_bn = nn.BatchNorm2d(planes[0])

        self.layer1 = self._make_layer(planes[0], layers[0])
        self.layer2 = self._make_layer(planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(planes[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bu_fc = nn.Linear(planes[3], num_classes)

        self.bu_features = self.bu_layer1_in = None
        self.td_conv1_in = None
        self.pre_pooling_size = None

        self.embedding = nn.Embedding(num_instructions, planes[3])
        self.td_fc = nn.Linear(2 * planes[3], planes[3])

        self.td_multp = nn.Parameter(torch.rand(planes[0]))
        self.td_conv1 = nn.Conv2d(planes[0], 1, kernel_size=7, padding=3, bias=False)
        self.td_bn1 = nn.BatchNorm2d(planes[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = upsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

            upsample = nn.Sequential(
                InterpolateConv(planes, self.inplanes, scale_factor=stride),
                nn.BatchNorm2d(self.inplanes)
            )

        layers = [CounterStreamBlock(self.inplanes, planes, stride, downsample, upsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(CounterStreamBlock(planes, planes))

        return nn.ModuleList(layers)

    def bottom_up(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.td_conv1_in is not None:
            x = self.td_conv1_in * self.bu_multp[None, :, None, None].expand_as(self.td_conv1_in) + x
            x = self.bu_side_bn(x)
            x = self.relu(x)

        self.bu_layer1_in = x

        for blk in self._iter_inner():
            x = blk(x, 'BU')

        self.pre_pooling_size = x.shape
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        self.bu_features = x
        x = self.bu_fc(x)

        return x

    def top_down(self, x):
        x = self.embedding(x)
        x = torch.cat((x, self.bu_features), dim=1)
        x = self.td_fc(x)
        x = self.relu(x)

        x = x[:, :, None, None].expand(self.pre_pooling_size)
        for blk in reversed(list(self._iter_inner())):
            x = blk(x, 'TD')

        x = self.bu_layer1_in * self.td_multp[None, :, None, None].expand_as(self.bu_layer1_in) + x
        x = self.td_bn1(x)
        x = self.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.td_conv1(x)
        return x

    def clear(self):
        self.bu_features = self.bu_layer1_in = self.td_conv1_in = self.pre_pooling_size = None
        for blk in self._iter_inner():
            blk.clear()

    def _iter_inner(self):
        iterator = itertools.chain(self.layer1, self.layer2, self.layer3, self.layer4)
        return iterator
