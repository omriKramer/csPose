# %%
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import coco_utils
from datasets import CocoSingleKPS
from mycoco import *

# %%

coco = CocoSingleKPS(root, ann_file, transform=T.ToTensor())
data_loader = DataLoader(coco, batch_size=4, num_workers=4, collate_fn=coco_utils.collate_fn)

# %%

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, progress=False)
model.eval()
print('Loaded Model')

# %%

it = iter(data_loader)

# %%
image, targets = next(it)
outputs = model(image)
#%%
orig_images = [coco.coco.get_img(t['image_id']) for t in targets]
to_tensor = T.ToTensor()
img_tensor = [to_tensor(oimg) for oimg in orig_images]
orig_outputs = model(img_tensor)

# %%
i = 2
coco_utils.show_image_with_kps(image[i], outputs[i]['keypoints'][0])

# %%
coco_utils.show_image_with_kps(orig_images[i], orig_outputs[i]['keypoints'][0])

#%%
coco.coco.show_by_id(targets[i]['image_id'])
#%%