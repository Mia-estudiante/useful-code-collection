import os
import numpy as np
from collections import OrderedDict
from PIL import Image as PILImage
from glob import glob

#%%
#SOLIDER-S  LIP
DSIZE = (572,384)
NUM_CLASSES = 4
DATASET = 'LIP'
DIR_GT = "/LIP/TrainVal_parsing_annotations/val_put_palette"
DIR_PRED = "/LIP/TrainVal_parsing_annotations/model_results/SOLIDER_SwinS_train"

a = compute_mean_ioU_LIP(DIR_PRED, DIR_GT, NUM_CLASSES, DATASET, DSIZE)
#%%
#SOLIDER-S  Anyang
DSIZE = (572,384)
NUM_CLASSES = 4
DATASET = 'LIP'
DIR_GT = "/parsing-reid/정리/Anyang/val_segmentations"
DIR_PRED = "/LIP/TrainVal_parsing_annotations/model_results/ReID/Anyang/SOLIDER_SwinS_train"
a = compute_mean_ioU_Anyang_Market1501(DIR_PRED, DIR_GT, NUM_CLASSES, DATASET, DSIZE)
#%%
#SOLIDER-S  Market1501
DSIZE = (572,384)
NUM_CLASSES = 4
DATASET = 'LIP'
DIR_GT = "/parsing-reid/정리/Market1501/val_segmentations"
DIR_PRED = "/LIP/TrainVal_parsing_annotations/model_results/ReID/Market1501/SOLIDER_SwinS_train"
a = compute_mean_ioU_Anyang_Market1501(DIR_PRED, DIR_GT, NUM_CLASSES, DATASET, DSIZE)
#%%
#SOLIDER-S  DukeMTMC
DSIZE = (572,384)
NUM_CLASSES = 4
DATASET = 'Duke'
DIR_GT = "/parsing-reid/정리/DukeMTMC/total/val_segmentations"
DIR_PRED = "/LIP/TrainVal_parsing_annotations/model_results/ReID/DukeMTMC/SOLIDER_SwinS_train"
a = compute_mean_ioU_Duke(DIR_PRED, DIR_GT, NUM_CLASSES, DATASET, DSIZE)
#%%
def compute_mean_ioU_LIP(dir_pred, dir_gt, num_classes=4, dataset='LIP', dsize=(473,473)):
    
    #Step1. confusion_matrix 초기화
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    #Step2. mapping할 dictionary 가져오기
    dic_map = make_dicmap(dataset)
    path_pr = sorted(glob(dir_pred+ '/*'))

    #Step3. 모델 predict한 값들을 resize해서 gt와 비교
    for pth_pr in path_pr:
        #1) PILImage로 열고 resize
        pred = PILImage.open(pth_pr).resize(dsize[::-1])                                        
        basename = os.path.basename(pth_pr)
        gt = PILImage.open(f'{dir_gt}/{basename}').resize(dsize[::-1])

        #2) 라벨링 값 서로 매핑
        gt2 = gt.point(lambda p: dic_map.get(p, 0))                                             
        pred2 = pred.point(lambda p: dic_map.get(p, 0))                     

        #3) np.array로 변환해서 confusion matrix 생성
        confusion_matrix += get_confusion_matrix(np.array(gt2), np.array(pred2), num_classes)   

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    if num_classes==4:
        CLASSES = ['Background', 'bottom', 'upper', 'dress']
    elif num_classes==20: 
        CLASSES = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', \
          'Right-leg', 'Left-shoe', 'Right-shoe']
    for i, (label, iou) in enumerate(zip(CLASSES, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value) 

    return name_value

#%%
'''
    market: / 하의 1 / 상의 2 / 드레스 3
    anyang: / 하의 1 / 상의 2 / 드레스 3
    => pred를 맞춰줘야 함
'''
def compute_mean_ioU_Anyang_Market1501(dir_pred, dir_gt, num_classes=4, dataset='LIP', dsize=(473,473)):
    
    #Step1-1. confusion_matrix 초기화
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    #Step1-2. mapping할 dictionary 가져오기 -> 나중에 다 맞춰보자
    dic_map = make_dicmap(dataset)
    path_pr = sorted(glob( dir_pred+ '/*'))

    #Step2. 모델 predict한 값들을 resize해서 gt와 비교
    for pth_pr in path_pr:
        #1) PILImage로 열고 resize
        pred = PILImage.open(pth_pr).resize(dsize[::-1])                    
        basename = os.path.basename(pth_pr)
        
        gt = PILImage.open(f'{dir_gt}/{basename}').resize(dsize[::-1])
        #2) 라벨링 값 서로 매핑 - gt의 경우, 이미 bkg, upper, bottom, dress로 잘 라벨링되어있므로 따로 할 필요 없다.
        pred_mapping = pred.point(lambda p: dic_map.get(p, 0))                     

        #3) np.array로 변환해서 confusion matrix 생성
        confusion_matrix += get_confusion_matrix(np.array(gt), np.array(pred_mapping), num_classes)   

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    if num_classes==4:
        CLASSES = ['Background', 'bottom', 'upper', 'dress']
    elif num_classes==20: 
        CLASSES = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', \
          'Right-leg', 'Left-shoe', 'Right-shoe']
        
    for i, (label, iou) in enumerate(zip(CLASSES, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value) 

    return name_value
#%%
def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32').reshape(-1)
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes)) 
    
    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix

#%%
def make_dicmap(dataset):
    dic2clt = {'upper':2, 'bottom':1, 'dress':3}

    dic_parset = {
        'ppss': {'upper':[3], 'bottom':[4]},  
        'LIP': {'upper':[5,7], 'bottom':[9,12], 'dress':[6,10]},
        'MHP_v2': {'upper':[10,11,12,13,14,15,34,57],  
                'bottom':[17,18,19,58],  
                'dress':[35,36,37,38]},  
        'Duke': {'upper':[3], 'bottom':[5], 'dress': [200]},
        'HPD': {'upper':[4,5], 'bottom':[6], 'dress':[7]}
    }

    dic_map = {v: dic2clt[key] for key, value in dic_parset[dataset].items() for v in value}
    return dic_map

#%%
'''
    duke: / 하의 5 / 상의 3 / 드레스 200
    => gt, pred를 맞춰줘야 함
'''
def compute_mean_ioU_Duke(dir_pred, dir_gt, num_classes=4, dataset='Duke', dsize=(473,473)):
    
    #Step1-1. confusion_matrix 초기화
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    #Step1-2. mapping할 dictionary 가져오기
    dic_map_pred = make_dicmap('LIP')
    dic_map_gt = make_dicmap(dataset)
    path_pr = sorted(glob(dir_pred+ '/*'))

    #Step2. 모델 predict한 값들을 resize해서 gt와 비교
    for pth_pr in path_pr:
        #1) PILImage로 열고 resize
        pred = PILImage.open(pth_pr).resize(dsize[::-1])                    
        basename = os.path.basename(pth_pr)
        gt = PILImage.open(f'{dir_gt}/{basename}').resize(dsize[::-1])

        #2-1) 라벨링 값 서로 매핑
        pred_mapping = pred.point(lambda p: dic_map_pred.get(p, 0))                     
        gt_mapping = gt.point(lambda p: dic_map_gt.get(p, 0))                     

        #3) np.array로 변환해서 confusion matrix 생성
        confusion_matrix += get_confusion_matrix(np.array(gt_mapping), np.array(pred_mapping), num_classes)   

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    if num_classes==4:
        CLASSES = ['Background', 'bottom', 'upper', 'dress']
    elif num_classes==20: 
        CLASSES = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', \
          'Right-leg', 'Left-shoe', 'Right-shoe']
        
    for i, (label, iou) in enumerate(zip(CLASSES, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value) 

    return name_value