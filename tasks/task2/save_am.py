'''
2024.01
acivaiont am
'''
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#%%
##global variables
ROOT = "/activation_map"
ROOT_ACTIVATE = os.path.join(ROOT, 'activate_map')

##save directory
SAVE_UPPER = os.path.join(ROOT, "upper")
SAVE_BOTTOM = os.path.join(ROOT, "bottom")
SAVE_BKG = os.path.join(ROOT, "bkg")
SAVE_DRESS = os.path.join(ROOT, "dress")

SAVE_ACTIVATE_UPPER = os.path.join(ROOT_ACTIVATE, "upper")
SAVE_ACTIVATE_BOTTOM = os.path.join(ROOT_ACTIVATE, "bottom")
SAVE_ACTIVATE_BKG = os.path.join(ROOT_ACTIVATE, "bkg")
SAVE_ACTIVATE_DRESS = os.path.join(ROOT_ACTIVATE, "dress")

##logits, image directory
LOGITS_PATH = os.path.join(ROOT, "activate_logits.npy")
IMAGE_NAMES_PATH = os.path.join(ROOT, "image_names.txt")
RAW_IMAGES_PATH = "/data/SOLIDER-HumanParsing/data/Market1501/val_images"
#%%
#function1. 특정 채널 별 activation map 추출
def make_activation_map(channels, log):
    part_am = log[:,50:-50,channels]    #1) padding된 것을 삭제하기 위함
    part_am = np.max(part_am, axis=-1)  #2) channles에서 가장 큰 값을 꼽기 위함
    plt.imshow(part_am, cmap='jet')     #3) 추출한 logits에 color map 입히기
    plt.show()

    return part_am
#%%
#function2. activation map 전처리
def preprocessing_am(part_am):
    part_am = cv2.resize(part_am, (128, 384)) # Resizing (384, 128)
    heatmapshow = cv2.normalize(part_am, None, alpha=0, beta=155, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_HSV)
    # heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

    return heatmapshow
#%%
#function3. raw image에 activation map 입혀서 저장
def save_images(img_name, output_img, output_am, cloth_type):
    output_img = Image.fromarray(np.asarray(output_img, dtype=np.uint8))
    output_am = Image.fromarray(np.asarray(output_am, dtype=np.uint8))

    if cloth_type=='upper':
        save_path1 = SAVE_UPPER
        save_path2 = SAVE_ACTIVATE_UPPER
    elif cloth_type=='bottom':
        save_path1 = SAVE_BOTTOM
        save_path2 = SAVE_ACTIVATE_BOTTOM
    elif cloth_type=='dress':
        save_path1 = SAVE_DRESS
        save_path2 = SAVE_ACTIVATE_DRESS
    elif cloth_type=='bkg':
        save_path1 = SAVE_BKG
        save_path2 = SAVE_ACTIVATE_BKG
        
    img_result_path = os.path.join(save_path1, img_name+".png")
    am_result_path = os.path.join(save_path2, img_name+".png")

    output_img.save(img_result_path)
    output_am.save(am_result_path)
#%%
#function4. activation map 저장
# def save_am():
#     output_im = Image.fromarray(np.asarray(output_img, dtype=np.uint8))
#     if cloth_type=='upper':
#         save_path = SAVE_UPPER
#     elif cloth_type=='bottom':
#         save_path = SAVE_BOTTOM
#     elif cloth_type=='dress':
#         save_path = SAVE_DRESS
#     elif cloth_type=='bkg':
#         save_path = SAVE_BKG
#     am_result_path = os.path.join(save_path, img_name+".png")
#     output_im.save(am_result_path)

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
    raw_image = cv2.resize(raw_image, (128, 384))       #384by128로 resize
    # raw_image = Image.fromarray(np.asarray(raw_image, dtype=np.uint8))

    #Step1. activation map 추출
    #1) 상의
    upper_am = make_activation_map(LIP["upper"], log)   #upper만 관련된 log만을 사용할 것
    #2) 하의
    bottom_am = make_activation_map(LIP['bottom'], log)
    #3) 드레스
    dress_am = make_activation_map(LIP['dress'], log)
    #4) 배경
    bkg_am = make_activation_map(LIP['bkg'], log)   

    #Step2. preprocessing 및 저장
    #1) 상의
    # upper_am_pp = preprocessing_am(upper_am)
    # upper_output_img = (raw_image*0.5 + 0.5*upper_am_pp).astype(np.uint8)
    # save_images(image_name, upper_output_img, upper_am_pp, 'upper')
    # #2) 하의
    # bottom_am_pp = preprocessing_am(bottom_am)
    # bottom_output_img = (raw_image*0.5 + 0.5*bottom_am_pp).astype(np.uint8)
    # save_images(image_name, bottom_output_img, bottom_am_pp, 'bottom')
    # #3) 드레스
    # dress_am_pp = preprocessing_am(dress_am)
    # dress_output_img = (raw_image*0.5 + 0.5*dress_am_pp).astype(np.uint8)
    # save_images(image_name, dress_output_img, dress_am_pp, 'dress')
    # #4) 배경
    # bkg_am_pp = preprocessing_am(bkg_am)
    # bkg_output_img = (raw_image*0.5 + 0.5*bkg_am_pp).astype(np.uint8)
    # save_images(image_name, bkg_output_img, bkg_am_pp, 'bkg')