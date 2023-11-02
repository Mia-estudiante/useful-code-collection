import os

#path 설정
ROOT = '/Market1501'
GALLERY = os.path.join(ROOT, 'bounding_box_test', 'val_images')
TRAIN = os.path.join(ROOT, 'bounding_box_train', 'val_images')
QUERY = os.path.join(ROOT, 'query', 'val_images')

#save path 설정
SAVE_ROOT = '/market'
SAVE_GALLERY = os.path.join(SAVE_ROOT, 'bounding_box_test')
SAVE_TRAIN = os.path.join(SAVE_ROOT, 'train')
SAVE_QUERY = os.path.join(SAVE_ROOT, 'query')

'''
function: list to txt
'''
def save_txt(save_path, save_list):
    save_list = list(map(lambda x: x[:-4], save_list))    
    with open(save_path, 'w+') as file:
        file.write('\n'.join(save_list))

save_txt(os.path.join(SAVE_GALLERY, 'val_id.txt'), os.listdir(GALLERY))
save_txt(os.path.join(SAVE_TRAIN, 'val_id.txt'), os.listdir(TRAIN))
save_txt(os.path.join(SAVE_QUERY, 'val_id.txt'), os.listdir(QUERY))
