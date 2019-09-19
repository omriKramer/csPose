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
from engine import eval


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='~/weizmann/coco/dev', help='dataset location')
    parser.add_argument('-d', '--device', default='cuda', choices=['cuda', 'cpu'], help='device')
    parser.add_argument('--num-workers', default=0, type=int, metavar='N', help='number of workers to use')
    parser.add_argument('-e', '--epochs', default=13, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--resume', default='')

    # output parameters
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--plot-freq', type=int, help='plot frequency')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args


def setup_output(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def load_from_checkpoint(checkpoint, model, map_location=None, optimizer=None):
    checkpoint = torch.load(checkpoint, map_location=map_location)
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


def default_model_feeder(model, images, _):
    return model(images)


def feed_images_and_targets(model, images, targets):
    return model(images, targets)


def get_train_msg(meters, iter_time, data_time, n_batch, epoch, i):
    n_spaces = len(str(n_batch))
    eta_seconds = iter_time.global_avg * (n_batch - i)
    eta = datetime.timedelta(seconds=int(eta_seconds))
    meters = meters_to_sting(meters)
    msg = f'Train - Epoch [{epoch}]: [{i:{n_spaces}d}/{n_batch}], eta: {eta}, {meters}, time: {iter_time}, data: {data_time}'
    if torch.cuda.is_available():
        MB = 1024.0 * 1024.0
        msg += f', max mem: {torch.cuda.max_memory_allocated() / MB:.4}'

    return msg


def meters_to_sting(meters):
    return ', '.join(f'{name}: {value:.4}' for name, value in meters.items())


def print_end_epoch(phase, data_loader, epoch, total_time):
    total_time_str = datetime.timedelta(seconds=int(total_time))
    print(f'{phase} - Epoch [{epoch}]: Total time: {total_time_str} ({total_time / len(data_loader):.4} s / it)')


class Engine:

    def __init__(self, model, data_path='.', output_dir='.', batch_size=32, device='cpu', epochs=1,
                 resume='', optimizer=None, model_feeder=None, num_workers=0, world_size=1,
                 dist_url='env://', print_freq=100, plot_freq=None):
        self.plot_freq = plot_freq if utils.is_main_process() else None
        self.print_freq = print_freq
        self.dist_url = dist_url
        self.world_size = world_size
        self.epochs = epochs
        self.data_path = data_path
        self.batch_size = batch_size
        self.output_dir = setup_output(output_dir)
        self.num_workers = num_workers
        self.start_epoch = 0
        self.optimizer = None

        device_index = self._init_distributed_mode()
        self.device = torch.device(f'{device}:{device_index}')

        self.model = model
        self.model.to(self.device)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device])

        params = [p for p in model.parameters() if p.requires_grad]
        if optimizer:
            self.optimizer = optimizer(params)

        if resume:
            load_from_checkpoint(resume, self.model, self.device, self.optimizer)

        if model_feeder:
            self.model_feeder = model_feeder
        else:
            self.model_feeder = default_model_feeder

        self.writer = SummaryWriter(self.output_dir)

    @classmethod
    def command_line_init(cls, model, **kwargs):
        args = get_args()
        engine = cls(model, **vars(args), **kwargs)
        return engine

    def run(self, train_ds, val_ds, evaluator, val_evaluator, loss_fn, collate_fn=None):
        print('Dataset Info')
        print('-' * 10)
        print(f'Train: {train_ds}')
        print()
        print(f'Validation: {val_ds}')
        print()

        print("Creating data loaders")
        train_loader, val_loader = self.create_loaders(train_ds, val_ds, collate_fn)

        print('Start training')
        start_time = time.time()
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            if self.distributed:
                train_loader.batch_sampler.sampler.set_epoch(epoch)

            self.train_one_epoch(train_loader, evaluator, epoch, loss_fn)
            self.evaluate(val_loader, val_evaluator, epoch)

            self.create_checkpoint(epoch)

        total_time = time.time() - start_time
        print('Done.')
        print(f'Total time {total_time // 60:.0f}m {total_time % 60:.0f}s')

    def train_one_epoch(self, data_loader, evaluator, epoch, loss_fn):
        self.model.train()
        evaluator.reset()

        start_time = time.time()
        end = time.time()
        iter_time = eval.SmoothedValue(fmt='{avg:.4f}')
        data_time = eval.SmoothedValue(fmt='{avg:.4f}')
        n_batch = len(data_loader)

        for i, (images, targets) in enumerate(data_loader):
            data_time.update(time.time() - end)
            images, targets = self.to_device(images, targets)
            outputs = self.model_feeder(self.model, images, targets)

            loss = loss_fn(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            evaluator.eval(targets, outputs, loss=loss, reduce=True)

            if i % self.print_freq == self.print_freq - 1:
                meters = evaluator.emit()
                print(get_train_msg(meters, iter_time, data_time, n_batch, epoch, i))
                self.write_scalars(meters, epoch, i, n_batch, name='train')

            iter_time.update(time.time() - end)
            end = time.time()

        total_time = time.time() - start_time
        print_end_epoch('Train', data_loader, epoch, total_time)
        print()

    @torch.no_grad()
    def evaluate(self, data_loader, evaluator, epoch):
        self.model.eval()
        start_time = time.time()

        for i, (images, targets) in enumerate(data_loader):
            images, targets = self.to_device(images, targets)
            outputs = self.model_feeder(self.model, images, targets)

            batch_results = evaluator.eval(targets, outputs)
            if self.plot_freq and i % self.plot_freq == self.plot_freq - 1:
                title, fig = evaluator.create_plots(batch_results, images, targets, outputs)
                self.writer.add_figure(title, fig, epoch)

        total_time = time.time() - start_time
        print_end_epoch('Val', data_loader, epoch, total_time)

        evaluator.synchronize_between_processes()
        meters = evaluator.emit()
        self.write_scalars(meters, epoch, name='val')
        print(meters_to_sting(meters))
        print()

    def to_device(self, images, targets):
        if torch.is_tensor(images):
            images = images.to(self.device)
        else:
            images = [img.to(self.device) for img in images]

        if isinstance(targets, dict):
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

    def create_checkpoint(self, epoch):
        if not utils.is_main_process():
            return

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.output_dir / f'checkpoint{epoch:03}.tar')

    def _init_distributed_mode(self):
        if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
            print('Not using distributed mode')
            self.distributed = False
            return 0

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])

        self.distributed = True
        self.dist_backend = 'nccl'
        print('| distributed init (rank {}): {}'.format(self.rank, self.dist_url), flush=True)
        dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                world_size=self.world_size, rank=self.rank)
        dist.barrier()
        utils.setup_for_distributed(self.rank == 0)
        return gpu

    def write_scalars(self, scalars, epoch, iteration=None, epoch_size=None, name=''):
        if iteration and epoch_size:
            global_step = epoch * epoch_size + iteration
        else:
            global_step = epoch

        if name:
            scalars = {f'{tag}/{name}': value for tag, value in scalars.items()}

        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, global_step)
