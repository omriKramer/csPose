import numbers

import torch
from fastai import vision as fv
from torch import nn
from torch.nn import functional as F


def block(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm2d(out_planes, out_planes),
        nn.ReLU()
    )


class TDHead(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm2d(64, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, num_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = x.squeeze(dim=1)
        return x


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 2d tensor.
    Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        kernel_size (int): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """

    def __init__(self, sigma, kernel_size=3, scale=True, thresh=0):
        super(GaussianSmoothing, self).__init__()

        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * 2

        # The gaussian kernel is the product of the  gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in (kernel_size, kernel_size)])

        for std, mgrid in zip(sigma, meshgrids):
            mean = (kernel_size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        if scale:
            # Make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)

        kernel = kernel.view(1, 1, *kernel.size())
        self.register_buffer('kernel', kernel)
        self.padding = kernel_size // 2
        self.thresh = thresh

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            x (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        channels = x.shape[1]
        weight = self.kernel.repeat(channels, *[1] * (self.kernel.dim() - 1))
        out = F.conv2d(x, weight, groups=channels, padding=self.padding)
        out[out < self.thresh] = 0
        out.clamp_(0, 1)
        return out


def conv_layer(ni, nf, ks=3):
    return nn.Sequential(
        fv.conv2d(ni, nf, ks=ks, bias=False),
        fv.batchnorm_2d(nf),
        nn.ReLU(inplace=True)
    )


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """Create and initialize a `nn.Conv1d` layer with spectral normalization."""
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return fv.spectral_norm(conv)


class PPMHead(nn.Sequential):

    def __init__(self, ni, nf, bin_size):
        super().__init__(
            nn.AdaptiveAvgPool2d(bin_size),
            *conv_layer(ni, nf)
        )


class PPM(nn.Module):

    def __init__(self, ni, bins=(1, 2, 3, 6)):
        super().__init__()
        heads = [PPMHead(ni, ni // len(bins), b) for b in bins]
        self.ppm_heads = nn.ModuleList(heads)
        self.conv = conv_layer(ni, ni)

    def forward(self, x):
        original_size = x.shape[2:]
        out = [head(x) for head in self.ppm_heads]
        out = [F.interpolate(o, size=original_size, mode='bilinear', align_corners=False) for o in out]
        out = torch.cat(out, dim=1)
        out = self.conv(out)
        return out
