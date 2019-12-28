import torch
from torch import nn


def feed_forward(model, x, instructions, td_head, skip_lateral):
    """
    out shape: b * n_instructions [* td_layers_out] * h * w
    """
    model.clear()
    td = []
    for inst in instructions:
        model(x, 'BU')

        td_out = model(inst, 'TD')
        td_out = td_head(td_out)
        td.append(td_out)

        if skip_lateral:
            model.clear()

    td = torch.stack(td, dim=1)
    return td


class SequentialInstructor(nn.Module):
    """
    Wraps a Counter Stream model inference to a single forward pass.
    Feeds in the instructions in a sequential manner.
    """

    def __init__(self, model, n_instructions, td_head, embedding=False, skip_lateral=False):
        super().__init__()
        if skip_lateral:
            model.one_iteration()

        self.skip_lateral = skip_lateral
        self.model = model
        self.td_head = td_head

        if embedding:
            instructions = torch.tensor(range(n_instructions), dtype=torch.long)
        else:
            instructions = torch.eye(n_instructions, dtype=torch.float)
        self.register_buffer('instructions', instructions)

    @property
    def name(self):
        name = f'{self.__class__.__name__}({self.model.name}, {len(self.instructions)})'
        return name

    def forward(self, x):
        batch_size = x.shape[0]
        instructions = self.instructions.expand(batch_size, *self.instructions.shape).transpose(0, 1)
        td = feed_forward(self.model, x, instructions, self.td_head, self.skip_lateral)
        return td

# class SimpleInstructor(nn.Module):
#     def __init__(self, model, td_head=None):
#         super().__init__()
#         self.model = model
#         self.td_head = td_head
#
#     @property
#     def name(self):
#         has_head = bool(self.td_head)
#         name = f'{self.__class__.__name__}({self.model.name}, head={has_head})'
#         return name
#
#     def forward(self, x, instructions):
#         td = feed_forward(self.model, x, instructions, self.td_head)
#         return td
