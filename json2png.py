'''
2023.12
(Human Parsing) dataset 마다 가지고 있는 annotation format 을 동일한 형태로 맞춰주기 위함
- visualization 함으로써 확인하는 데 용이하도록 함
( * 각 데이터 맞춤 전처리 코드는 mybranch 내 'KIST-SOLIDER-HumanParsing/preprocess/crop'에 있음을 참고 )
'''

import os
import json
import numpy as np
from PIL import ImageDraw
from PIL import Image as PILImage
import matplotlib.pyplot as plt

def get_palette(data):
    PALETTE = {
        "anydataset": [0,0,0,               #Background        
                        0,128,0,              #Short sleeve top
                        0,128,0,              #Long sleeve top
                        0,128,0,              #Short sleeve outwear
                        0,128,0,              #Long sleeve outwear
                        0,128,0,              #Vest
                        0,128,0,              #Sling
                        128,0,0,              #Shorts
                        128,0,0,              #Trousers
                        128,0,0,              #Skirt
                        0,0,85,               #Short sleeve dress
                        0,0,85,               #Long sleeve dress
                        0,0,85,               #Vest dress 
                        0,0,85,               #Sling dress
                    ]
    }

    return PALETTE[data]

'''
json 파일로 annotation 되어 있는 데이터셋의 gt를 png 파일로 만들기 위함
'''
def json2png(root, data, json_root, images_path, save_name):

    # Step1. make a json list
    json_list = [ json_name for json_name in os.listdir(json_root) ]

    # data 에 맞는 palette 불러오기 
    palette = get_palette(data)

    # define the name of save directory
    segmentation_path = os.path.join(root, save_name)
    if not os.path.exists(segmentation_path):
        os.makedirs(segmentation_path)

    for j in json_list:
        with open(os.path.join(json_root, j)) as f:
            json_data = json.load(f)

        img_name = j.split('.')[0]
        raw_path = os.path.join(images_path, f'{img_name}.jpg')
        raw_image = PILImage.open(raw_path)

        mask = PILImage.new('L', raw_image.size, 0)
        draw = ImageDraw.Draw(mask)

        # Stpe2. draw segmentations - matplotlib
        for item, values in json_data.items():
        # dict_keys(['item2', 'source', 'pair_id', 'item1'])
        # item{n} 은 하나의 카테고리를 의미
            if 'item' not in item: continue

            id = values['category_id']
            seg = values['segmentation']

            if len(seg)==1:
                s = seg
                segmentation_np = np.array(s).reshape((-1, 2)).astype(int)
                draw.polygon(segmentation_np.flatten().tolist(), outline=id, fill=id)
            else:
                for s in seg:
                    segmentation_np = np.array(s).reshape((-1, 2)).astype(int)
                    draw.polygon(segmentation_np.flatten().tolist(), outline=id, fill=id)

        result = PILImage.fromarray(np.asarray(mask, dtype=np.uint8))
        result.putpalette(palette)
        result.save(os.path.join(segmentation_path, f'{img_name}.png'))