'''
2023.05
raw image 에 특정 카테고리에 맞게 mask 를 씌우기 위함
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

'''
raw image, gt, pred 순서로 array 형식으로 맞춰주기 위함
'''
def trans2arr(name):
    jpg_name = images[0][:-4]+'.jpg'
    png_name = images[0][:-4]+'.png'

    #1. images, gt, model_result
    img_path1 = os.path.join(IMAGES, jpg_name)
    img_path2 = os.path.join(GT, png_name)
    img_path3 = os.path.join(MODEL_RESULT, png_name)

    #2. images, gt, model_result: np array
    jpg_out = Image.open(img_path1)
    jpg_arr = np.array(jpg_out)

    gt_out = Image.open(img_path2)
    gt_arr = np.array(gt_out)

    model_out = Image.open(img_path3)
    model_arr = np.array(model_out)

    return jpg_arr, gt_arr, model_arr

'''
save PIL Image from array
'''
def save_arr2pil(arr, target, name):
    img = Image.fromarray(arr)                         
    img.save(f'{os.path.join(target, name)}.jpg', 'JPEG')

IMAGES = '/RAW_IMAGES'
GT = '/GT'
MODEL_RESULT = '/MODEL_RESULT'

img_name = 'img_name'

images = os.listdir(IMAGES)
gt_results = os.listdir(GT)
model_results = os.listdir(MODEL_RESULT)

# Step 1. raw image, gt, pred array
jpg_arr, gt_arr, model_arr = trans2arr(img_name) 

# raw image에 mask를 씌울 array들
jpg_arr_gt = jpg_arr.copy()
jpg_arr_model = jpg_arr.copy()

# Step 2-1. gt mask 씌우기 - 'upper':[5,7]
jpg_arr_gt[..., 0][gt_arr==5] = 255
jpg_arr_gt[..., 0][gt_arr==7] = 255
plt.imshow(jpg_arr_gt)
save_arr2pil(jpg_arr_gt, '저장 root 폴더', img_name)

# Step 2-2. pred mask 씌우기 - 'upper':[5,7]
if len(jpg_arr_model.shape)==3:                   # Case1. 컬러 이미지의 경우
    jpg_arr_model[..., 0][model_arr==5] = 255
    jpg_arr_model[..., 0][model_arr==7] = 255
else:                                             # Case2. 흑백 이미지의 경우, 컬러 이미지와 동일하게 차원을 3개로 맞춰줌
    r, g, b = np.expand_dims(jpg_arr_model.copy(), axis=2), np.expand_dims(jpg_arr_model.copy(), axis=2), np.expand_dims(jpg_arr_model.copy(), axis=2)
    jpg_arr_model = np.concatenate((r, g, b),axis=2)

    jpg_arr_model[..., 0][model_arr==5] = 255
    jpg_arr_model[..., 0][model_arr==7] = 255
plt.imshow(jpg_arr_model)
save_arr2pil(jpg_arr_model, '저장 root 폴더', img_name)

# Extra. 겹치지 않는 이미지 찾기
jpg_arr_model_x = jpg_arr.copy()
jpg_arr_model_x[jpg_arr_model != jpg_arr_gt] = 255