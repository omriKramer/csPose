import torch
import torchvision
from torch.utils.data import DataLoader

import coco_utils
import transform
import engine.engine as eng

#%%
composed = transform.Compose([transform.ResizeKPS((80, 150)), transform.ToTensor()])
coco_val = eng.get_dataset('~/weizmann/coco', train=False, transforms=composed)
print('Dataset Info')
print('-' * 10)
print(f'Validation: {coco_val}')
print()

val_loader = DataLoader(coco_val, batch_size=4)

#%%
device = torch.device('cpu')
model = torchvision.models.resnet50(progress=False, num_classes=3 * len(coco_utils.KEYPOINTS))
epoch = eng.load_from_checkpoint('results/simple_resnet50/checkpoint039.tar', model, device)

#%%
data_iter = iter(val_loader)
images, targets = next(data_iter)

#%%
coco_val.coco.show_by_id(targets['id'][0].item())
