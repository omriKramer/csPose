import torch
from torch import nn


class SequentialInstructor(nn.Module):
    """
    Wraps a Counter Stream model inference to a single forward pass.
    Feeds in the instructions in a sequential manner.
    """

    def __init__(self, model, n_instructions):
        super().__init__()
        self.model = model
        self.register_buffer('instructions', torch.tensor(range(n_instructions), dtype=torch.long))

    def forward(self, x):
        """
        out shape: b * n_instructions [* td_layers_out] * h * w
        """
        self.model.clear()

        batch_size = x.shape[0]
        td = []
        for inst in self.instructions:
            self.model(x, 'BU')
            inst = inst.expand(batch_size)
            td.append(self.model(inst, 'TD'))
        td = torch.stack(td, dim=1)
        return td
