import math
import sys
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

import coco_utils
import engine.engine
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

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_oks = running_oks / len(data_loader.dataset)

        time_elapsed = time.time() - start_time
        phase = 'Train' if train else 'Val'
        print(f'{phase} Loss: {epoch_loss:.4f}, OKS: {epoch_oks:.4f}')
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()


def main(args):
    checkpoint_dir = setup_output(args.output_dir)

    composed = transform.Compose([transform.ResizeKPS((80, 150)), transform.ToTensor()])
    coco_train = engine.engine.get_dataset(args.data_path, train=True, transforms=composed)
    coco_val = engine.engine.get_dataset(args.data_path, train=False, transforms=composed)

    batch_size = args.batch_size * args.num_gpu
    train_loader = DataLoader(coco_train, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(coco_val, batch_size=batch_size, num_workers=4)

    model = torchvision.models.resnet50(progress=False, num_classes=3 * len(coco_utils.KEYPOINTS))
    device = torch.device(args.device)
    model.to(device)
    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    start_time = time.time()
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        print('-' * 10)
        one_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        one_epoch(model, val_loader, criterion, device)

    total_time = time.time() - start_time
    print(f'Total time {total_time // 60:.0f}m {total_time % 60:.0f}s')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_dir / f'checkpoint{args.epochs - 1:04}')


def setup_output(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


if __name__ == '__main__':
    main(engine.engine.get_args())
