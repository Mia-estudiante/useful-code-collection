'''
2024.01
여러 사람이 있는 이미지를 한 사람씩 쪼갬
'''
import os
import numpy as np
from PIL import Image as PILImage

'''
person patch 이미지를 instance id 에 맞게 crop 하기 위함
'''
def location(ins_id, ins_array):
    up_x, down_x, left_y, right_y = -1, -1, -1, -1

    for row_idx, row in enumerate(ins_array):
        if (up_x==-1) and (ins_id in row): up_x = row_idx
        if ins_id in row: down_x = row_idx
        for col_idx, col in enumerate(row):
            if (left_y==-1) and (ins_id==col): left_y = col_idx
            if (ins_id==col) and (col_idx<left_y): left_y = col_idx
            if (ins_id==col) and (right_y<col_idx): right_y = col_idx

    return up_x, down_x, left_y, right_y

'''
한 이미지에 다수가 존재하면서, instance id 의 annotation 이 존재하는 경우 사용

root
ㅣ- raw_name: raw images 폴더명
ㅣ- ins_name: instance id 폴더명
ㅣ- cate_name: mask images 폴더명
ㅣ- save_name: crop 한 raw image & mask image 를 저장할 폴더명
'''
def preprocess_multi_person_datasets(root, dataset, raw_name, ins_name, cate_name, save_name):
    
    #저장 폴더명
    segmentation_path = os.path.join(root, save_name, 'Segmentations')
    images_path = os.path.join(root, save_name, 'Images')

    if not os.path.exists(segmentation_path):
        os.makedirs(segmentation_path)
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    #이미지명
    text_path = os.path.join(root, f"{dataset}_id.txt")
    if os.path.exists(text_path):
        raw_list = [i_id.strip() for i_id in open(text_path)]
    else: 
        raw_list = [i_id.strip() for i_id in os.listdir(os.path.join(root, raw_name))]

    for name in raw_list:
        raw_image = os.path.join(root, raw_name, name+".jpg")
        ins_image = os.path.join(root, ins_name, name+".png")
        cate_image = os.path.join(root, cate_name, name+".png")

        raw_array = np.array(PILImage.open(raw_image))
        ins_array = np.array(PILImage.open(ins_image))
        ins_list = np.unique(ins_array)
        cate_array = np.array(PILImage.open(cate_image))

        for ins_id in ins_list[1:]:
            up_x, down_x, left_y, right_y = location(ins_id, ins_array)

            segmentation_result = np.where(ins_array[up_x:down_x,left_y:right_y]==ins_id, cate_array[up_x:down_x,left_y:right_y], 0)
            segmentation_result_pil = PILImage.fromarray(segmentation_result)

            raw_result = raw_array[up_x:down_x,left_y:right_y]
            raw_result_pil = PILImage.fromarray(raw_result)

            segmentation_result_pil.save(os.path.join(segmentation_path, name+f'_{ins_id}'+'.png'))
            raw_result_pil.save(os.path.join(images_path, name+f'_{ins_id}'+'.jpg'))