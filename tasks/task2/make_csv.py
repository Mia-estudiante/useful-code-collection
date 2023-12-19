'''
2023.05
원하는 속성 토대로 csv 파일 생성
'''
#%%
import os
import numpy as np
import pandas as pd
from more_itertools import locate

ROOT = '/home'
TEST_PATH = os.path.join(ROOT, 'bounding_box_test')
TRAIN_PATH = os.path.join(ROOT, 'bounding_box_train')
QUERY_PATH = os.path.join(ROOT, 'query')

#jpg인지 png인지 확인 필요할 것 
#fn_name: PID_CAMID_SEQUENCE_
query_images = os.listdir(QUERY_PATH)

bbox_tests = os.listdir(TEST_PATH)          #파일명 list -> 0005_c5s1_012676_01 (PID_CAMID_SEQUENCE_?)
bbox_trains = os.listdir(TRAIN_PATH)        #파일명 list -> 0005_c5s1_012676_01 (PID_CAMID_SEQUENCE_?)
test_PIDs = [bbox_test.split('_')[0] for bbox_test in bbox_tests] 
train_PIDs = [bbox_train.split('_')[0] for bbox_train in bbox_trains]

print(len(query_images))
# %%
count = 0
for query_image in query_images:
    try:
        count +=1
        print(count, query_image)

        test_data = list()              #fn_name, PID, CAMID, (score) 
        train_data = list()

        csv_name = query_image.split('.')[0]+'.csv'

        PID = query_image.split('_')[0]
        same_test_PIDs = list(locate(test_PIDs, lambda x: x==PID))      #PID와 동일한 test_PID의 인덱스 값으로 이뤄진 list
        same_train_PIDs = list(locate(train_PIDs, lambda x: x==PID))
        
        test_fn_name_list = [bbox_tests[i] for i in same_test_PIDs]
        test_PID_list = [PID]*len(test_fn_name_list)
        test_CAMID_list = [test_fn_name.split('_')[1] for test_fn_name in test_fn_name_list] 

        train_fn_name_list = [bbox_trains[i] for i in same_train_PIDs]
        train_PID_list = [PID]*len(train_fn_name_list)
        train_CAMID_list = [train_fn_name('_')[1] for train_fn_name in train_fn_name_list] 
        
        test_data.append(test_fn_name_list)
        test_data.append(test_PID_list)
        test_data.append(test_CAMID_list)

        train_data.append(train_fn_name_list)
        train_data.append(train_PID_list)
        train_data.append(train_CAMID_list)

        columns=['fn_name', 'PID', 'CAMID']
        test_dict = {
            columns[0]: test_data[0],
            columns[1]: test_data[1],
            columns[2]: test_data[2]
        }
        test_df = pd.DataFrame(data=test_dict)
        test_df.to_csv(ROOT+'/test/'+csv_name, index=False)

        train_dict = {
            columns[0]: train_data[0],
            columns[1]: train_data[1],
            columns[2]: train_data[2]
        }
        train_df = pd.DataFrame(data=train_dict)
        train_df.to_csv(ROOT+'/train/'+csv_name, index=False)
    except:
        # print(query_image+"에서 에러 발생!!!")
        pass