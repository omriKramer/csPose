import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import CocoSingleKPS


def get_dataset(data_path, train=True, transform=None, target_transform=None, transforms=None):
    data_path = Path(data_path).expanduser()
    image_set = 'train' if train else 'val'
    root = data_path / '{}2017'.format(image_set)
    ann_file = data_path / 'annotations/person_keypoints_{}2017.json'.format(image_set)
    return CocoSingleKPS(root, ann_file, transform=transform, target_transform=target_transform, transforms=transforms)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='~/weizmann/coco/dev', help='dataset location')
    parser.add_argument('-e', '--epochs', default=13, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-d', '--device', default='cuda', choices=['cuda', 'cpu'], help='device')
    parser.add_argument('-g', '--num-gpu', default=1, type=int, metavar='N', help='number of GPUs to use')
    parser.add_argument('--num-workers', default=0, type=int, metavar='N', help='number of workers to use')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='')

    args = parser.parse_args()
    return args


def setup_output(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def load_from_checkpoint(checkpoint, model, map_location=None, optimizer=None):
    checkpoint = torch.load(checkpoint, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


def write_metrics(writer, metrics, global_step):
    for metric, value in metrics.items():
        writer.add_scalar(metric, value, global_step)


def default_model_feeder(model, images, _):
    return model(images)


def feed_images_and_targets(model, images, targets):
    return model(images, targets)


class Engine:

    def __init__(self, model, data_path='.', output_dir='.', batch_size=32, device='cpu', epochs=1,
                 num_gpu=1, resume='', optimizer=None, model_feeder=None, num_workers=0):
        self.epochs = epochs
        self.data_path = data_path
        self.num_gpu = num_gpu
        self.batch_size = batch_size
        self.output_dir = setup_output(output_dir)
        self.device = torch.device(device)
        self.num_workers = num_workers

        model.to(device)
        if optimizer:
            self.optimizer = optimizer(model.parameters())

        self.start_epoch = 0
        if resume:
            self.start_epoch = load_from_checkpoint(resume, model, self.device, optimizer=self.optimizer)

        if self.num_gpu > 1:
            self.model = nn.DataParallel(model, device_ids=list(range(self.num_gpu)))
        else:
            self.model = model

        if model_feeder:
            self.model_feeder = model_feeder
        else:
            self.model_feeder = default_model_feeder

        self.train_writer = SummaryWriter(self.output_dir / 'train')
        self.val_writer = SummaryWriter(self.output_dir / 'test')

    @classmethod
    def command_line_init(cls, model, **kwargs):
        args = get_args()
        engine = cls(model, **vars(args), **kwargs)
        return engine

    def run(self, train_ds, val_ds, metrics, val_metrics=None, collate_fn=None):
        if not val_metrics:
            val_metrics = metrics

        print('Dataset Info')
        print('-' * 10)
        print(f'Train: {train_ds}')
        print()
        print(f'Validation: {val_ds}')
        print()

        batch_size = self.batch_size * self.num_gpu
        train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=self.num_workers, collate_fn=collate_fn,
                                  shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=self.num_workers, collate_fn=collate_fn)

        start_time = time.time()
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            print(f'Epoch {epoch}')
            print('-' * 10)

            train_results = self.one_epoch(train_loader, metrics, train=True)
            val_results = self.one_epoch(val_loader, val_metrics)

            write_metrics(self.train_writer, train_results, epoch)
            write_metrics(self.val_writer, val_results, epoch)
            self.create_checkpoint(epoch, train_results, val_results)

        total_time = time.time() - start_time
        print(f'Total time {total_time // 60:.0f}m {total_time % 60:.0f}s')

    def one_epoch(self, data_loader, metrics, train=False):
        if train:
            self.model.train()
        else:
            self.model.eval()

        running_metrics = Counter()
        start_time = time.time()

        for images, targets in data_loader:
            images, targets = self.to_device(images, targets)

            with torch.set_grad_enabled(train):
                outputs = self.model_feeder(self.model, images, targets)
                metric_values = {name: metric(outputs, targets) for name, metric in metrics.items()}

                for name, value in metric_values.items():
                    if not torch.isfinite(value):
                        print(f'{name} is {value}, stopping training')
                        sys.exit(1)

                running_metrics += metric_values
                if train:
                    loss = metric_values['loss']
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        time_elapsed = time.time() - start_time
        epoch_metrics = {name: value.item() / len(data_loader.dataset) for name, value in running_metrics.items()}

        phase = 'Train' if train else 'Val'
        metric_string = ', '.join((f'{name}: {value}' for name, value in epoch_metrics.items()))
        print(f'{phase} - {metric_string}')
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()

        return epoch_metrics

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

    def create_checkpoint(self, epoch, train_metrics, val_metrics):
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, self.output_dir / f'checkpoint{epoch:03}.tar')

    def get_dataset(self, train=True, transform=None, target_transform=None, transforms=None):
        return get_dataset(self.data_path, train=train, transform=transform, target_transform=target_transform,
                           transforms=transforms)
