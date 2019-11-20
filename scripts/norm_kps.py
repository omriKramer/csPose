from torchvision import transforms as T

from datasets import CocoSingleKPS
from utils import dataset_mean_and_std

transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
dataset = CocoSingleKPS.from_data_path('/Volumes/waic/shared/coco', train=True, transform=transform)

mean, std = dataset_mean_and_std(dataset)

print(f'mean: {mean:.4}')
print(f'std: {std:.4}')
