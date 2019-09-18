import datetime
import time
from collections import deque, Counter

import torch

import utils

MB = 1024.0 * 1024.0


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.deque.clear()

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, metrics, print_freq, epoch=0, name=''):
        self.header = name
        self.metrics = metrics
        self.meters = None
        self.epoch = epoch
        self.print_freq = print_freq

    def update(self, **kwargs):
        results = {k: v.item() if torch.is_tensor(v) else v for k, v in kwargs.items()}
        self.meters.update(results)

    def eval(self, targets, outputs):
        batch_results = self.metrics(targets, outputs)
        batch_metrics_reduced = utils.reduce_dict(batch_results)
        self.update(**batch_metrics_reduced)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        if self.meters is None:
            return 'No metrics were logged'

        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter / self.print_freq))
            )
        return ' '.join(loss_str)

    def iteration_header(self):
        epoch_header = f'Epoch: [{self.epoch}]'
        if self.name:
            header = f'{self.name} - {epoch_header}'
        else:
            header = epoch_header
        return header

    def iter_and_log(self, iterable):
        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        self.meters = Counter()
        header = self.iteration_header()
        nspace = str(len(str(len(iterable))))

        i = 0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if self.print_freq and i % self.print_freq == self.print_freq - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta = datetime.timedelta(seconds=int(eta_seconds))

                msg = f'{header} [{i:{nspace}d}/{len(iterable)}] eta: {eta} {self} time: {iter_time} data: {data_time}'
                if torch.cuda.is_available():
                    msg += f' max mem: {torch.cuda.max_memory_allocated() / MB:0f}'

                print(msg)
                self.meters = Counter()

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')

        # print meters only once in the end of the epoch
        if self.print_freq is None:
            print(self)
            print()

        self.epoch += 1
