import argparse
from pathlib import Path

import torch
from torch import nn

from datasets import CocoSingleKPS

SMOOTH = 1e-6


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


def create_checkpoint(path, model, optimizer, epoch, train_loss, val_metrics):
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_loss,
        'val_metrics': val_metrics,
    }, path / f'checkpoint{epoch:03}.tar')


def write_metrics(writer, metrics, global_step):
    for metric, value in metrics.items():
        writer.add_scalar(metric, value, global_step)