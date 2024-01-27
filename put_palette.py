'''
2024.01
원하는 palette를 씌워 mask image 획득
'''
import os
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt

def get_lip_palette():

    palette = [
            0,0,0,           #Background
            128,0,0,            #Hat
            255,0,0,            #Hair
            0,85,0,             #Glove
            170,0,51,           #Sunglasses
            255,85,0,           #Upper-clothes
            0,0,85,             #Dress
            0,119,221,          #Coat
            85,85,0,            #Socks
            0,85,85,            #Pants
            85,51,0,            #Jumpsuits
            52,86,128,          #Scarf
            0,128,0,            #Skirt
            0,0,255,            #Face
            51,170,221,         #Left-arm
            0,255,255,          #Right-arm
            85,255,170,         #Left-leg
            170,255,85,         #Right-leg
            255,255,0,          #Left-shoe
            255,170,0           #Right-shoe
    ]

    return palette

ROOT = 'ROOT'
img = 'image.png'
img_path = os.path.join(ROOT, img)
out = PILImage.open(img_path)

'''
< ValueError: unknown raw mode for given image mode >

out = PILImage.open(img_path).convert('P')
'''

out.putpalette(get_lip_palette())
plt.imshow(out)

out_pil = np.array(out)

'''
background 만 추출하고 싶은 경우

plt.imshow(out_pil==0)
'''
