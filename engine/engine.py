import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from coco_eval import CocoEval
from datasets import CocoSingleKPS


def get_dataset(data_path, train=True, transform=None, target_transform=None, transforms=None):
    data_path = Path(data_path).expanduser()
    image_set = 'train' if train else 'val'
    root = data_path / '{}2017'.format(image_set)
    ann_file = data_path / 'annotations/person_keypoints_{}2017.json'.format(image_set)
    return CocoSingleKPS(root, ann_file, transform=transform, target_transform=target_transform, transforms=transforms)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/Volumes/waic/shared/coco', help='dataset location')
    parser.add_argument('-e', '--epochs', default=13, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-d', '--device', default='cuda', choices=['cuda', 'cpu'], help='device')
    parser.add_argument('-g', '--num-gpu', default=1, type=int, metavar='N', help='number of GPUs to use')
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


evaluator = CocoEval()


class Engine:

    def __init__(self, model, data_path='.', output_dir='.', batch_size=32, device='cpu', epochs=10, num_gpu=1,
                 resume='',
                 optimizer=None):
        self.epochs = epochs
        self.data_path = data_path
        self.num_gpu = num_gpu
        self.batch_size = batch_size
        self.output_dir = setup_output(output_dir)
        self.device = torch.device(device)

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

        self.train_writer = SummaryWriter(self.output_dir / 'train')
        self.val_writer = SummaryWriter(self.output_dir / 'test')

    @classmethod
    def from_command_line(cls, model, optimizer=None):
        args = get_args()
        engine = cls(model, **vars(args), optimizer=optimizer)
        return engine

    def run(self, transforms=None):

        coco_train = get_dataset(self.data_path, train=True, transforms=transforms)
        coco_val = get_dataset(self.data_path, train=False, transforms=transforms)
        print('Dataset Info')
        print('-' * 10)
        print(f'Train: {coco_train}')
        print(f'Validation: {coco_val}')
        print()

        batch_size = self.batch_size * self.num_gpu
        train_loader = DataLoader(coco_train, batch_size=batch_size, num_workers=4, shuffle=True)
        val_loader = DataLoader(coco_val, batch_size=batch_size, num_workers=4)

        criterion = nn.MSELoss()

        start_time = time.time()
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            print(f'Epoch {epoch}')
            print('-' * 10)

            train_metrics = self.one_epoch(train_loader, criterion, train=True)
            val_metrics = self.one_epoch(val_loader, criterion)

            write_metrics(self.train_writer, train_metrics, epoch)
            write_metrics(self.val_writer, val_metrics, epoch)
            self.create_checkpoint(epoch, train_metrics, val_metrics)

        total_time = time.time() - start_time
        print(f'Total time {total_time // 60:.0f}m {total_time % 60:.0f}s')

    def one_epoch(self, data_loader, criterion, train=False):
        if train:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_oks = 0.0

        start_time = time.time()
        for images, targets in data_loader:
            images = images.to(self.device)
            keypoints = targets['keypoints'].to(self.device)
            areas = targets['area'].to(self.device)

            with torch.set_grad_enabled(train):
                outputs = self.model(images)

                loss = criterion(outputs, keypoints)
                if not math.isfinite(loss):
                    print("Loss is {}, stopping training".format(loss))
                    sys.exit(1)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item()
            for gt, dt, area in zip(keypoints, outputs, areas):
                running_oks += evaluator.compute_oks(gt, dt, area, device=self.device).item()

        time_elapsed = time.time() - start_time
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_oks = running_oks / len(data_loader.dataset)
        phase = 'Train' if train else 'Val'
        print(f'{phase} Loss: {epoch_loss:.4f}, OKS: {epoch_oks:.4f}')
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()

        return {
            'loss': epoch_loss,
            'oks': epoch_oks,
        }

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
