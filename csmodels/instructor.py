import torch
from torch import nn


class SequentialInstructor(nn.Module):
    """
    Wraps a Counter Stream model inference to a single forward pass.
    Feeds in the instructions in a sequential manner.
    """

    def __init__(self, model, instructions):
        super().__init__()
        self.model = model
        self.instructions = instructions

    def forward(self, x):
        self.model.clear()

        batch_size = x.shape[0]
        td = []
        for inst in self.instructions:
            self.model(x, 'BU')
            inst = torch.full((batch_size,), inst, dtype=torch.long)
            td.append(self.model(inst, 'TD'))

        td = torch.stack(td, dim=1)
        return td
