from collections import deque, defaultdict

import torch
import torch.distributed as dist

import utils


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

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

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


class MetricLogger:
    def __init__(self, metrics, plot_fn=None):
        self.create_plots = plot_fn
        self.metrics = metrics
        self.meters = defaultdict(lambda: SmoothedValue(window_size=None, fmt="{median:.4f}"))

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def eval(self, targets, outputs, loss=None, reduce=False):
        batch_results = self.metrics(targets, outputs)
        averaged_results = {name: values.mean() for name, values in batch_results.items()}
        if loss:
            averaged_results['loss'] = loss

        size = len(targets)
        if reduce:
            averaged_results = utils.reduce_dict(averaged_results)
            size *= utils.get_world_size()

        self.update(n=size, **averaged_results)
        return batch_results

    def emit(self):
        meters = {name: meter.avg for name, meter in self.meters.items()}
        self.reset()
        return meters

    def reset(self):
        self.meters.clear()

    def synchronize_between_processes(self):
        for key in sorted(self.meters.keys()):
            self.meters[key].synchronize_between_processes()
