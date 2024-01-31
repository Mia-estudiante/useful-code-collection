'''
2024.01
특정 idx를 갖는 mask image를 찾고자 함
'''
import os
import numpy as np
from glob import glob
from PIL import Image as PILImage

root = 'val_segmentations'
paths = sorted(glob(root+'/*'))
tmp = []
dress = []

for img_path in paths:
    out = PILImage.open(img_path)
    tmp += np.unique(out).tolist()

    if idx in np.unique(out):
        dress.append(img_path.split('/')[-1:])