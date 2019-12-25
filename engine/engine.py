import argparse
import datetime
import os
import time
from pathlib import Path

import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from engine import metric_logger


def get_args(args=None):
    """
    Example of using with DistributedDataParallel
    python3 -m torch.distributed.launch --nproc_per_node=NUM_GPU --use_env /path/to/scripty.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda', choices=['cuda', 'cpu'], help='device')
    parser.add_argument('--num-workers', default=4, type=int, metavar='N', help='number of workers to use')
    parser.add_argument('-e', '--epochs', default=13, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per process, the total batch size is $processes x batch_size')
    parser.add_argument('--resume', default='')

    # optimization parameters
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')

    # output parameters
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--print-freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--plot-freq', type=int, help='plot frequency in epochs')
    parser.add_argument('--out-file', action='store_true', help='output to a file instead of stdout')
    parser.add_argument('--flush', action='store_true')

    # distributed training parameters
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--data-parallel', action='store_true', help='use DataParallel')

    return parser.parse_args(args)


def load_from_checkpoint(checkpoint, model, map_location=None, optimizer=None, lr_scheduler=None):
    checkpoint = torch.load(checkpoint, map_location=map_location)
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


MB = 1024.0 * 1024.0


def get_train_msg(meters, iter_time, data_time, n_batch, epoch, i):
    n_spaces = len(str(n_batch))
    eta_seconds = iter_time.global_avg * (n_batch - i)
    eta = datetime.timedelta(seconds=int(eta_seconds))
    meters = meters_to_string(meters)
    msg = (f'Train - Epoch [{epoch}]: [{i:{n_spaces}d}/{n_batch}], eta: {eta},'
           f' {meters}, time: {iter_time}, data: {data_time}')
    if torch.cuda.is_available():
        msg += f', max mem: {torch.cuda.max_memory_allocated() / MB:.4f}'

    return msg


def meters_to_string(meters):
    return ', '.join(f'{name}: {value:.4f}' for name, value in meters.items())


def end_epoch_msg(phase, data_loader, epoch, total_time):
    total_time_str = datetime.timedelta(seconds=int(total_time))
    return f'{phase} - Epoch [{epoch}]: Total time: {total_time_str} ({total_time / len(data_loader):.2f} s / it)'


def infer_checkpoint(output_dir: Path):
    k = len('checkpoint')
    checkpoints = [child for child in output_dir.iterdir() if child.name.startswith('checkpoint')]
    if len(checkpoints) == 0:
        return None

    latest = max(checkpoints, key=lambda file: int(file.name[k:k + 3]))
    return latest


class Engine:

    def __init__(self, lr, momentum, weight_decay, lr_steps, lr_gamma,
                 data_path='.', output_dir='.', out_file=False, flush=False, batch_size=32, device='cpu', epochs=1,
                 resume='', num_workers=4, dist_url='env://', print_freq=100,
                 plot_freq=None, data_parallel=False, ):
        self._setup_output(output_dir, out_file, flush)

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_steps = lr_steps
        self.lr_gamma = lr_gamma

        self.plot_freq = plot_freq if utils.is_main_process() else None
        self.print_freq = print_freq

        self.epochs = epochs
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.start_epoch = 0

        self.dist_url = dist_url
        device_index = self._init_distributed_mode()
        self.device = torch.device(f'{device}:{device_index}')
        self.data_parallel = data_parallel
        assert not (self.data_parallel and self.distributed), 'use either DataParallel or DistributedDataParallel'

        if resume == 'auto':
            self.checkpoint = infer_checkpoint(self.output_dir)
        else:
            self.checkpoint = resume

    @classmethod
    def command_line_init(cls, args=None, **kwargs):
        args = get_args(args=args)
        engine = cls(**vars(args), **kwargs)
        return engine

    def run(self, model, train_ds, val_ds, evaluator, collate_fn=None, plot_fn=None):
        title = '| Engine Run started |'
        self.print('-' * len(title))
        self.print(title)
        self.print('-' * len(title))
        self.print()

        self.print('Dataset Info')
        self.print('-' * 10)
        self.print(f'Train: {train_ds}')
        self.print()
        self.print(f'Validation: {val_ds}')
        self.print()

        if isinstance(evaluator, torch.nn.Module):
            evaluator.to(self.device)
        model = self.setup_model(model)
        optimizer, lr_scheduler = self.setup_optimizer(model)

        self.print(f'Training info')
        self.print('-' * 10)
        self.print(self)
        self.print()
        if self.checkpoint:
            self.print(f'Loading from checkpoint {self.checkpoint}')
            self.start_epoch = load_from_checkpoint(self.checkpoint, model, self.device, optimizer, lr_scheduler)
        else:
            self.record_hparams()

        train_loader, val_loader = self.create_loaders(train_ds, val_ds, collate_fn)

        self.print('Start training...')
        start_time = time.time()
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            if self.distributed:
                train_loader.batch_sampler.sampler.set_epoch(epoch)

            self.train_one_epoch(model, optimizer, train_loader, evaluator, epoch)
            meters = self.evaluate(model, val_loader, evaluator, epoch, plot_fn)
            iterations = (epoch + 1) * len(train_ds)
            self.write_scalars(meters, iterations, name='val')
            lr_scheduler.step()
            self.create_checkpoint(model, optimizer, epoch, lr_scheduler)

        total_time = time.time() - start_time
        self.print('Done.')
        self.print(f'Total time {total_time // 60:.0f}m {total_time % 60:.0f}s')

        if utils.is_main_process():
            if self.out_file:
                self.out_file.close()
            self.writer.close()

    def train_one_epoch(self, model, optimizer, data_loader, evaluator, epoch):
        model.train()
        logger = metric_logger.MetricLogger()
        start_time = time.time()
        end = time.time()
        iter_time = metric_logger.SmoothedValue(fmt='{avg:.4f}')
        data_time = metric_logger.SmoothedValue(fmt='{avg:.4f}')

        for i, (images, targets) in enumerate(data_loader):
            data_time.update(time.time() - end)

            optimizer.zero_grad()
            images, targets = self.to_device(images, targets)
            outputs = model(images)
            batch_results, _ = evaluator(outputs, targets)
            loss = batch_results['loss']
            assert torch.isfinite(loss), f'Loss is {loss} on epoch {epoch} iter {i}'
            loss.backward()
            optimizer.step()

            logger.update(batch_results, len(images), reduce=True)
            if i % self.print_freq == self.print_freq - 1:
                meters = logger.emit()
                meters['lr'] = optimizer.param_groups[0]["lr"]
                self.print(get_train_msg(meters, iter_time, data_time, len(data_loader), epoch, i))
                iterations = epoch * len(data_loader.dataset) + self.batch_size * self.world_size * (i + 1)
                self.write_scalars(meters, iterations, name='train')

            iter_time.update(time.time() - end)
            end = time.time()

        total_time = time.time() - start_time
        self.print(end_epoch_msg('Train', data_loader, epoch, total_time))
        self.print()

    @torch.no_grad()
    def evaluate(self, model, data_loader, evaluator, epoch, plot_fn=None):
        model.eval()
        logger = metric_logger.MetricLogger()
        start_time = time.time()

        for i, (images, targets) in enumerate(data_loader):
            images, targets = self.to_device(images, targets)
            outputs = model(images)

            batch_results, preds = evaluator(outputs, targets)
            logger.update(batch_results, len(images))
            if plot_fn and self._should_plot(epoch, i, len(data_loader)):
                title, fig = plot_fn(batch_results, images, targets, preds)
                title += f'/{i}'
                self.writer.add_figure(title, fig, epoch)

        total_time = time.time() - start_time
        self.print(end_epoch_msg('Val', data_loader, epoch, total_time))

        logger.synchronize_between_processes(self.device)
        meters = logger.emit()
        self.print(meters_to_string(meters))
        self.print()
        return meters

    def _should_plot(self, epoch, iteration, total_iterations):
        if not utils.is_main_process() or not self.plot_freq:
            return False

        if epoch % self.plot_freq == 0 or epoch == self.start_epoch + self.epochs - 1:
            period = max(total_iterations // 10, 1)
            if iteration % period == 0:
                return True

        return False

    def setup_model(self, model):
        self.print('Model')
        self.print('-' * 10)

        self.print(model.name)
        self.print()
        if self.data_parallel:
            model = nn.DataParallel(model)
            self.print("Using DataParallel with", torch.cuda.device_count(), "GPUs")
            model.to(self.device)
        elif self.distributed:
            model.to(self.device)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device])
        else:
            model.to(self.device)

        return model

    def setup_optimizer(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_steps, gamma=self.lr_gamma)

        return optimizer, lr_scheduler

    def to_device(self, images, targets):
        if torch.is_tensor(images):
            images = images.to(self.device)
        else:
            images = [img.to(self.device) for img in images]

        if torch.is_tensor(targets):
            targets = targets.to(self.device)
        elif isinstance(targets, dict):
            targets = {k: v.to(self.device) for k, v in targets.items()}
        else:
            targets = [{k: v.to(self.device) for k, v in d.items()} for d in targets]

        return images, targets

    def create_loaders(self, train_ds, val_ds, collate_fn):
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_ds)
            val_sampler = torch.utils.data.SequentialSampler(val_ds)

        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, self.batch_size, drop_last=True)
        train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, num_workers=self.num_workers,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, sampler=val_sampler, num_workers=self.num_workers,
                                collate_fn=collate_fn)

        return train_loader, val_loader

    def create_checkpoint(self, model, optimizer, epoch, lr_scheduler):
        if not utils.is_main_process():
            return

        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }, self.output_dir / f'checkpoint{epoch:03}.tar')

    def _init_distributed_mode(self):
        if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
            self.print('Not using distributed mode')
            self.distributed = False
            self.world_size = 1
            return 0

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)

        self.distributed = True
        self.dist_backend = 'nccl'
        self.print('| distributed init (rank {}): {}'.format(self.rank, self.dist_url), flush=True)
        dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                world_size=self.world_size, rank=self.rank)
        dist.barrier()
        utils.setup_for_distributed(self.rank == 0)
        return gpu

    def write_scalars(self, scalars, global_step, name=''):
        if not utils.is_main_process():
            return

        if name:
            scalars = {f'{tag}/{name}': value for tag, value in scalars.items()}

        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, global_step)

    def __repr__(self):
        d = {
            'lr': self.lr,
            'momentum': self.momentum,
            'weight decay': self.weight_decay,
            'lr steps': self.lr_steps,
            'gamma': self.lr_gamma,
            'batch size': self.batch_size,
            'epochs': self.epochs,
            'world_size': self.world_size,
        }

        r = ', '.join(f'{name}: {value}' for name, value in d.items())
        return r

    def record_hparams(self, metrics=None):
        if not metrics:
            metrics = {}

        if not utils.is_main_process():
            return

        hparams_dict = {
            'optimizer': 'SGD',
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'gamma': self.lr_gamma,
            'bsize': self.batch_size * self.world_size,
        }
        if self.lr_steps:
            hparams_dict['lr_steps'] = ', '.join(str(s) for s in self.lr_steps)

        metrics_dict = {f'hparam/{name}': value for name, value in metrics.items()}
        self.writer.add_hparams(hparams_dict, metrics_dict)

    def print(self, *objects):
        print(*objects, file=self.out_file, flush=self.flush)

    def _setup_output(self, output_dir, out_file, flush):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flush = flush
        self.out_file = None
        if utils.is_main_process():
            if out_file:
                self.out_file = (self.output_dir / 'train.txt').open('a')

            self.writer = SummaryWriter(output_dir)
