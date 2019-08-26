# %%
import matplotlib.pyplot as plt
import numpy as np

from datasets import CocoSingleKPS
# %%
root = '../coco/val2017/'
ann_file = '../coco/annotations/person_keypoints_val2017.json'

coco = CocoSingleKPS(root, ann_file)
# %%
img, kps = coco[10]

#%%
plt.imshow(np.asarray(img))
plt.show()

#%%
coco.show_item(10)
plt.show()

#%%
from pycocotools.coco import COCO
import skimage.io as io

coco = COCO(ann_file)
cats = coco.loadCats(coco.getCatIds())
img = coco.loadImgs([32817])[0]
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()
#%%

