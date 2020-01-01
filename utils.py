import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def my_print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = my_print


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='~/weizmann/coco/dev', help='dataset location')
    parser.add_argument('--lose', default='ce', choices=['ce', 'kl'], help='which lose to use')
    args, remaining_args = parser.parse_known_args()
    return args, remaining_args


def dataset_mean_and_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0.
    for i, (images, _) in enumerate(loader):
        batch_samples = images.shape[0]
        data = images.reshape(batch_samples, images.shape[1], -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std
