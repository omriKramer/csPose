import torch
from torch import nn

from .cs_resnet import resnet18
from .layers import TDHead


class CSBaseline(nn.Module):
    def __init__(self, n_channels=17):
        super().__init__()
        self.backbone = resnet18(td_outplanes=64, num_instructions=1)
        self.td_head = TDHead(num_channels=n_channels)
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
