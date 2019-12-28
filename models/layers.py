from torch import nn


def block(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm2d(out_planes, out_planes),
        nn.ReLU()
    )


class TDHead(nn.Module):

    def __init__(self):
        super(TDHead, self).__init__()
        self.head = nn.Sequential(
            nn.BatchNorm2d(64, 64),
            nn.ReLU(),
            block(64, 64),
            block(64, 32, stride=2),
            block(32, 32),
            block(32, 16, stride=2),
            block(16, 16),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.head(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x.squeeze(dim=1)
        return x
