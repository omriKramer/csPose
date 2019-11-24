import argparse

from torchvision import transforms as T

from datasets import CocoSingleKPS
from utils import dataset_mean_and_std

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str)
args = parser.parse_args()

transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
dataset = CocoSingleKPS.from_data_path(args.data_path, train=True, transform=transform)

mean, std = dataset_mean_and_std(dataset)

print(f'mean: {mean}')
print(f'std: {std}')
