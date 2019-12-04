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
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda', choices=['cuda', 'cpu'], help='device')
    parser.add_argument('--num-workers', default=4, type=int, metavar='N', help='number of workers to use')
    parser.add_argument('-e', '--epochs', default=13, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
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
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--plot-freq', type=int, help='plot frequency in epochs')
    parser.add_argument('--overwrite', action='store_true', help='delete contents of output dir before running')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser.parse_args(args)


def setup_output(output_dir, overwrite=False):
    output_dir = Path(output_dir)
    if overwrite and output_dir.is_dir():
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()

    return output_dir


def load_from_checkpoint(checkpoint, model, map_location=None, optimizer=None, lr_scheduler=None):
    checkpoint = torch.load(checkpoint, map_location=map_location)
    if isinstance(model, nn.parallel.DistributedDataParallel):
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


def print_end_epoch(phase, data_loader, epoch, total_time):
    total_time_str = datetime.timedelta(seconds=int(total_time))
    print(f'{phase} - Epoch [{epoch}]: Total time: {total_time_str} ({total_time / len(data_loader):.2f} s / it)')


def infer_checkpoint(output_dir: Path):
    k = len('checkpoint')
    checkpoints = [child for child in output_dir.iterdir() if child.name.startswith('checkpoint')]
    if len(checkpoints) == 0:
        return None

    latest = max(checkpoints, key=lambda file: int(file.name[k:k + 3]))
    return latest


class Engine:

    def __init__(self, lr, momentum, weight_decay, lr_steps, lr_gamma,
                 data_path='.', output_dir='.', batch_size=32, device='cpu', epochs=1,
                 resume='', num_workers=4, world_size=1,
                 dist_url='env://', print_freq=100, plot_freq=None, overwrite=False, ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_steps = lr_steps
        self.lr_gamma = lr_gamma

        self.plot_freq = plot_freq if utils.is_main_process() else None
        self.print_freq = print_freq

        self.dist_url = dist_url
        self.world_size = world_size

        self.epochs = epochs
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.start_epoch = 0

        device_index = self._init_distributed_mode()
        self.device = torch.device(f'{device}:{device_index}')

        self.output_dir = setup_output(output_dir, overwrite=overwrite and utils.is_main_process())
        if utils.is_main_process():
            self.writer = SummaryWriter(output_dir)

        if resume == 'auto':
            self.checkpoint = infer_checkpoint(self.output_dir)
        else:
            self.checkpoint = resume

    @classmethod
    def command_line_init(cls, args=None, **kwargs):
        args = get_args(args=args)
        engine = cls(**vars(args), **kwargs)
        return engine

    def run(self, model, train_ds, val_ds, evaluator, val_evaluator, collate_fn=None, plot_fn=None):
        print('Dataset Info')
        print('-' * 10)
        print(f'Train: {train_ds}')
        print()
        print(f'Validation: {val_ds}')
        print()

        model = self.setup_model(model)
        optimizer, lr_scheduler = self.setup_optimizer(model)
        print(f'Training info:')
        print(self)
        if self.checkpoint:
            print(f'Loading from checkpoint {self.checkpoint}')
            self.start_epoch = load_from_checkpoint(self.checkpoint, model, self.device, optimizer, lr_scheduler)
        train_loader, val_loader = self.create_loaders(train_ds, val_ds, collate_fn)

        print('Start training...')
        start_time = time.time()
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            if self.distributed:
                train_loader.batch_sampler.sampler.set_epoch(epoch)

            self.train_one_epoch(model, optimizer, train_loader, evaluator, epoch)
            self.evaluate(model, val_loader, val_evaluator, epoch, plot_fn)
            lr_scheduler.step()
            self.create_checkpoint(model, optimizer, epoch, lr_scheduler)

        total_time = time.time() - start_time
        print('Done.')
        print(f'Total time {total_time // 60:.0f}m {total_time % 60:.0f}s')

        if utils.is_main_process():
            self.writer.flush()

    def train_one_epoch(self, model, optimizer, data_loader, evaluator, epoch):
        model.train()
        logger = metric_logger.MetricLogger()
        start_time = time.time()
        end = time.time()
        iter_time = metric_logger.SmoothedValue(fmt='{avg:.4f}')
        data_time = metric_logger.SmoothedValue(fmt='{avg:.4f}')
        n_batch = len(data_loader)

        for i, (images, targets) in enumerate(data_loader):
            data_time.update(time.time() - end)

            optimizer.zero_grad()
            images, targets = self.to_device(images, targets)
            outputs = model(images)
            batch_results = evaluator(outputs, targets)
            loss = batch_results['loss']
            assert torch.isfinite(loss), f'Loss is {loss} on epoch {epoch} iter {i}'
            loss.backward()
            optimizer.step()

            logger.update(batch_results, len(images), reduce=True)
            if i % self.print_freq == self.print_freq - 1:
                meters = logger.emit()
                meters['lr'] = optimizer.param_groups[0]['lr']
                print(get_train_msg(meters, iter_time, data_time, n_batch, epoch, i))
                self.write_scalars(meters, epoch, i, n_batch, name='train')

            iter_time.update(time.time() - end)
            end = time.time()

        total_time = time.time() - start_time
        print_end_epoch('Train', data_loader, epoch, total_time)
        print()

    @torch.no_grad()
    def evaluate(self, model, data_loader, evaluator, epoch, plot_fn=None):
        model.eval()
        logger = metric_logger.MetricLogger()
        start_time = time.time()

        for i, (images, targets) in enumerate(data_loader):
            images, targets = self.to_device(images, targets)
            outputs = model(images)

            batch_results = evaluator(outputs, targets)
            logger.update(batch_results, len(images))
            if plot_fn and self._should_plot(epoch, i, len(data_loader)):
                title, fig = plot_fn(batch_results, images, targets, outputs)
                title += f'/{i}'
                self.writer.add_figure(title, fig, epoch)

        total_time = time.time() - start_time
        print_end_epoch('Val', data_loader, epoch, total_time)

        logger.synchronize_between_processes(self.device)
        meters = logger.emit()
        self.write_scalars(meters, epoch, name='val')
        print(meters_to_string(meters))
        print()

    def _should_plot(self, epoch, iteration, total_iterations):
        if not utils.is_main_process() or not self.plot_freq:
            return False

        if epoch % self.plot_freq == 0 or epoch == self.start_epoch + self.epochs - 1:
            period = max(total_iterations // 10, 1)
            if iteration % period == 0:
                return True

        return False

    def setup_model(self, model):
        print('setup mode...')
        model.to(self.device)
        if self.distributed:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device])

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
        print("Creating data loaders")
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

        if isinstance(model, nn.parallel.DistributedDataParallel):
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
            print('Not using distributed mode')
            self.distributed = False
            return 0

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)

        self.distributed = True
        self.dist_backend = 'nccl'
        print('| distributed init (rank {}): {}'.format(self.rank, self.dist_url), flush=True)
        dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                world_size=self.world_size, rank=self.rank)
        dist.barrier()
        utils.setup_for_distributed(self.rank == 0)
        return gpu

    def write_scalars(self, scalars, epoch, iteration=None, epoch_size=None, name=''):
        if not utils.is_main_process():
            return

        if iteration and epoch_size:
            global_step = epoch * epoch_size + iteration
        else:
            global_step = epoch

        if name:
            scalars = {f'{tag}/{name}': value for tag, value in scalars.items()}

        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, global_step)

    def __repr__(self):
        d = {
            'optimization':
                {
                    'lr': self.lr,
                    'momentum': self.momentum,
                    'weight decay': self.weight_decay,
                    'lr steps': self.lr_steps,
                    'gamma': self.lr_gamma
                },
            'batch size': self.batch_size,
            'epochs': self.epochs,
            'world_size': self.world_size,
        }

        if self.checkpoint:
            d['checkpoint'] = self.checkpoint

        return repr(d)
