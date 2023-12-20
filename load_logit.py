#%%
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#%%
# global variables
ROOT = "/SOLIDER_base/results"

# directories to save
SAVE_UPPER = os.path.join(ROOT, "upper")
SAVE_BOTTOM = os.path.join(ROOT, "bottom")
SAVE_BKG = os.path.join(ROOT, "bkg")
SAVE_DRESS = os.path.join(ROOT, "dress")

# logits, image names, image directory
LOGITS_PATH = os.path.join(ROOT, "activate_logits.npy")
IMAGE_NAMES_PATH = os.path.join(ROOT, "image_names.txt")
RAW_IMAGES_PATH = "/raw_images"
#%%
'''
특정 채널 별 activation map 을 추출하기 위함
'''
def make_activation_map(channels, log):
    part_am = log[:,50:-50,channels]
    part_am = np.max(part_am, axis=-1)
    # plt.imshow(part_am, cmap='jet')
    # plt.show()

    return part_am
#%%
'''
activation map 전처리
'''
def preprocessing_am(part_am):
    part_am = cv2.resize(part_am, (128, 384)) # Resizing (384, 128)
    heatmapshow = cv2.normalize(part_am, None, alpha=0, beta=155, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_HSV)
    # heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

    return heatmapshow
#%%
'''
raw image 에 class 별 activation map 씌워 저장
'''
def save_add_am_raw(img_name, output_img, cloth_type):
    output_im = Image.fromarray(np.asarray(output_img, dtype=np.uint8))
    if cloth_type=='upper':
        save_path = SAVE_UPPER
    elif cloth_type=='bottom':
        save_path = SAVE_BOTTOM
    elif cloth_type=='dress':
        save_path = SAVE_DRESS
    elif cloth_type=='bkg':
        save_path = SAVE_BKG
    am_result_path = os.path.join(save_path, img_name+".png")
    output_im.save(am_result_path)
#%%
#Step1-1. 이미지명 로드
image_names = []
with open(IMAGE_NAMES_PATH, "r") as file:
    for i in file:
        image_names.append(i.strip())
print(image_names)
#%%
#Step1-2. logits 로드
logits = np.load(LOGITS_PATH)
logits = logits[1:, ]
#%%
#Step2. logit 뽑기 & raw 이미지에 적용
LIP = {'bkg': [0], 'upper':[5,7], 'bottom':[9,12], 'dress':[6,10]}

for idx, (image_name, log) in enumerate(zip(image_names, logits)):
    
    #raw_image 설정
    raw_image_path = os.path.join(RAW_IMAGES_PATH, image_name+'.jpg')
    raw_image = Image.open(raw_image_path)
    raw_image = np.asarray(raw_image, dtype=np.uint8)
    raw_image = cv2.resize(raw_image, (128, 384))
    # raw_image = Image.fromarray(np.asarray(raw_image, dtype=np.uint8))

    #Step2-1. activation map 추출
    #1) 상의
    upper_am = make_activation_map(LIP["upper"], log)
    #2) 하의
    bottom_am = make_activation_map(LIP['bottom'], log)
    #3) 드레스
    dress_am = make_activation_map(LIP['dress'], log)
    #4) 배경
    bkg_am = make_activation_map(LIP['bkg'], log)   

    #Step2-2. preprocessing
    #1) 상의
    upper_am_pp = preprocessing_am(upper_am)
    upper_output_img = (raw_image*0.5 + 0.5*upper_am_pp).astype(np.uint8)
    save_add_am_raw(image_name, upper_output_img, 'upper')
    #2) 하의
    bottom_am_pp = preprocessing_am(bottom_am)
    bottom_output_img = (raw_image*0.5 + 0.5*bottom_am_pp).astype(np.uint8)
    save_add_am_raw(image_name, bottom_output_img, 'bottom')
    #3) 드레스
    dress_am_pp = preprocessing_am(dress_am)
    dress_output_img = (raw_image*0.5 + 0.5*dress_am_pp).astype(np.uint8)
    save_add_am_raw(image_name, dress_output_img, 'dress')
    #4) 배경
    bkg_am_pp = preprocessing_am(bkg_am)
    bkg_output_img = (raw_image*0.5 + 0.5*bkg_am_pp).astype(np.uint8)
    save_add_am_raw(image_name, bkg_output_img, 'bkg')