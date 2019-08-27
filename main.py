import matplotlib.pyplot as plt
import numpy as np

from datasets import CocoSingleKPS
# %%
from transform import resize_image_and_keypoints

# %%
root = '../coco/val2017/'
ann_file = '../coco/annotations/person_keypoints_val2017.json'

coco = CocoSingleKPS(root, ann_file)
coco_transform = CocoSingleKPS(root, ann_file, transofrms=(
    lambda img, kps: resize_image_and_keypoints(img, kps, (1280, 480))))
# %%
item = 1554
coco.show_item(item)
plt.show()

# %%
coco_transform.show_item(item)
plt.show()

# %%
sizes = [img.size for img, _ in coco]

# %%
import itertools

max_size = max(itertools.chain.from_iterable(sizes))
min_size = min(itertools.chain.from_iterable(sizes))

# %%
for i, size in enumerate(sizes):
    if size[0] == min_size or size[1] == min_size:
        print(i)
        break
# %%
ann = coco.annotations[1554]
# %%
from pycocotools.coco import COCO
import skimage.io as io

coco = COCO(ann_file)
cats = coco.loadCats(coco.getCatIds())
# %%
img = coco.loadImgs([11197])[0]
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
plt.show()
# %%
