from torch import nn


class SequentialInstructor(nn.Module):
    """
    Wraps a Counter Stream model inference to a single forward pass.
    Feeds in the instructions in a sequential manner.
    """

    def __init__(self, model, instructions):
        super().__init__()
        self.model = model
        self.instruction = instructions

    def forward(self, x):
        out = {'bu': [], 'td': []}
        batch_size = x.shape[0]
        for inst in self.instruction:
            inst = inst.expand(batch_size)
            self.model(x, 'BU')
            out['td'].append(self.model(inst, 'TD'))

        return out
