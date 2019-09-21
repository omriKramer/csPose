from abc import ABC, abstractmethod

from torch import nn


class CSBlock(nn.Module, ABC):

    def forward(self, x, mode):
        if mode == 'BU':
            return self._bottom_up(x)
        elif mode == 'TD':
            return self._top_down(x)

        raise ValueError(f'mode must be "TD" or "BU" got {mode}')

    @abstractmethod
    def _bottom_up(self, x):
        pass

    @abstractmethod
    def _top_down(self, x):
        pass

    @abstractmethod
    def clear(self):
        pass
