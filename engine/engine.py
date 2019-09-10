import argparse
from pathlib import Path

import torch

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
    parser.add_argument('--output-dir', default='/home/labs/waic/omrik', help='path where to save')
    parser.add_argument('--resume', default='')

    args = parser.parse_args()
    return args


def setup_output(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


def load_from_checkpoint(checkpoint, device, model, optimizer=None):
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    return start_epoch
