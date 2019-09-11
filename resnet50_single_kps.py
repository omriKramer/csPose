import math
import sys
import time

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import coco_utils
import engine.engine as eng
import transform
from coco_eval import CocoEval

evaluator = CocoEval()


def one_epoch(model, data_loader, criterion, device, optimizer=None):
    train = optimizer is not None
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_oks = 0.0

    start_time = time.time()
    for images, targets in data_loader:
        images = images.to(device)
        keypoints = targets['keypoints'].to(device)
        areas = targets['area'].to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)

            loss = criterion(outputs, keypoints)
            if not math.isfinite(loss):
                print("Loss is {}, stopping training".format(loss))
                sys.exit(1)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        for gt, dt, area in zip(keypoints, outputs, areas):
            running_oks += evaluator.compute_oks(gt, dt, area, device=device).item()

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


def main(args):
    output_dir = eng.setup_output(args.output_dir)
    train_writer = SummaryWriter(output_dir / 'train')
    val_writer = SummaryWriter(output_dir / 'test')

    composed = transform.Compose([transform.ResizeKPS((80, 150)), transform.ToTensor()])
    coco_train = eng.get_dataset(args.data_path, train=True, transforms=composed)
    coco_val = eng.get_dataset(args.data_path, train=False, transforms=composed)
    print('Dataset Info')
    print('-' * 10)
    print(f'Train: {coco_train}')
    print(f'Validation: {coco_val}')
    print()

    batch_size = args.batch_size * args.num_gpu
    train_loader = DataLoader(coco_train, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(coco_val, batch_size=batch_size, num_workers=4)

    device = torch.device(args.device)
    model = torchvision.models.resnet50(progress=False, num_classes=3 * len(coco_utils.KEYPOINTS))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    start_epoch = 0
    if args.resume:
        print(f'Loading checkpoint {args.resume}')
        start_epoch = eng.load_from_checkpoint(args.resume, model, device, optimizer)

    if args.num_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    end_epoch = start_epoch + args.epochs
    start_time = time.time()
    for epoch in range(start_epoch, end_epoch):
        print(f'Epoch {epoch}')
        print('-' * 10)

        train_metrics = one_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = one_epoch(model, val_loader, criterion, device)

        eng.write_metrics(train_writer, train_metrics, epoch)
        eng.write_metrics(val_writer, val_metrics, epoch)
        eng.create_checkpoint(output_dir, model, optimizer, epoch, train_metrics, val_metrics)

    total_time = time.time() - start_time
    print(f'Total time {total_time // 60:.0f}m {total_time % 60:.0f}s')


if __name__ == '__main__':
    main(eng.get_args())
