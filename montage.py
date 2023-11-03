import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import montage

#raw images
IMAGES = '/LIP/TrainVal_images/TrainVal_images/val_images'

#BOTTOM
GT_BOTTOM = '/bottom/gt'
MODEL_RESULT_BOTTOM = '/bottom/hhp_model'
MODEL473_RESULT_BOTTOM = '/bottom/hhp_model_473'

#UPPER
GT_UPPER = '/upper/gt'
MODEL_RESULT_UPPER = '/upper/hhp_model'
MODEL473_RESULT_UPPER = '/upper/hhp_model_473'

#DRESS
GT_DRESS = '/dress/gt'
MODEL_RESULT_DRESS = '/dress/hhp_model'
MODEL473_RESULT_DRESS = '/dress/hhp_model_473'

#모아서 뿌려줄 list
ccc = []

DSIZE = (30,20)
names = []

# gray = []
gray_file = []

NOW = MODEL473_RESULT_DRESS #핸들링하는 곳
COUNT = 100                 #이미지 개수

result_name = NOW.split("/")[-1]            #gt, model명
TARGET = '/plot_images/'+NOW.split("/")[-2] #저장할 폴더

for file in os.listdir(IMAGES)[:COUNT]:
    filename = os.path.join(NOW, file)
    # out.save(os.path.join(TARGET, filename))

    img = Image.open(filename).resize(DSIZE[::-1])

    if np.array(img).shape != (30, 20, 3): #흑백 사진의 경우, 패스
        # gray.append(np.array(img))
        gray_file.append(file)

    ccc.append(np.array(img))
    names.append(file)

out = montage(ccc, channel_axis=-1)
out = Image.fromarray(out)
out.save(os.path.join(TARGET, f'{TARGET.split("/")[-1]}_{result_name}_{COUNT}.png'))