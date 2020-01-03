import torch
from torch import nn

from .cs_resnet import resnet18


class BaselineHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm2d(64, 64),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 17, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return x


class CSBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(td_outplanes=64, num_instructions=1)
        self.td_head = BaselineHead()
        self.backbone.one_iteration()
        instruction = torch.ones(1, dtype=torch.float)
        self.register_buffer('instruction', instruction)
        self.name = 'CSBaseline18'

    def forward(self, x):
        instruction = self.instruction.expand(x.shape[0], 1)
        self.backbone.clear()
        self.backbone(x, 'BU')
        td_out = self.backbone(instruction, 'TD')
        td_out = self.td_head(td_out)
        return td_out
