'''
2022.12
특정 object 개수에 맞춰 각각의 디렉토리에 저장
'''
import os 
import json
import random

def check(goal, future_value):
    if future_value < goal:
        return 0
    elif future_value == goal:
        return 1
    else: 
        return 2

def check_full_list(now_labels, full_list):
    for label in now_labels:
        if label in full_list:
            return False
    else:
        return True

def is_two(future_dict):
    for key in future_dict.keys(): 
        if future_dict[key][1]==2:
            return False
    return True

'''
[ 현재 세고 있는 라벨 개수 / 원하는 목표 라벨 개수 ] 
labels_count[클래스][0] = 현재 세고 있는 라벨 개수
labels_count[클래스][1] = 원하는 목표 라벨 개수
'''
labels_count = {
    "can": [0, 5400],           #통조림
    "clothes": [0, 3600],       #의류
    "fishing_plumb": [0, 5400], #낚시추
    "glass_bottle": [0, 7200],  #유리병
    "gloves": [0, 5400],        #장갑
    "lure": [0, 5400],          #루어
    "mask": [0, 6000],          #마스크
    "mixed": [0, 0],            
    "plastic_bottle": [0, 9000],#페트병
    "scrap_iron": [0, 7200],    #고철 
    "soda_can": [0, 5400]       #음료수캔
}

# 1. 현재 디렉토리에서 json파일만 추출
file_path = "/json" # json 경로

file_list = os.listdir(file_path)
file_list = [[file_name] for file_name in file_list if file_name.endswith(".json") ]

# 원하는 object 개수가 다 채워질 때까지 json 파일 저장
extracted_list = list()

# 목표치 채워진 클래스들
full_list = list()

# 2. 현재 이미지 annotation 내 객체 개수 추출
for sub_list in file_list:
    json_path = file_path + "/" + sub_list[0]
    with open(json_path, 'r') as file:
        data = json.load(file)
        sub_list.append(len(data["shapes"]))

        temp = {
            "can": 0,
            "clothes": 0,
            "fishing_plumb": 0,
            "glass_bottle": 0,
            "gloves": 0,
            "lure": 0,
            "mask": 0,
            "mixed": 0,
            "plastic_bottle": 0,
            "scrap_iron": 0,
            "soda_can": 0
        }

        for i in range(len(data["shapes"])):
            label = data["shapes"][i]["label"]
            temp[label] += 1

        for key in list(temp.keys()):
            if temp[key]!=0:
                continue
            del temp[key]
        sub_list.append(temp)

# 3. 객체 개수 기준으로 역정렬
file_list.sort(key=lambda x : x[1], reverse=True)

only1 = [sub_list for sub_list in file_list if sub_list[1] == 1]    # annotation 개수가 1개인 것
remain = [sub_list for sub_list in file_list if sub_list[1] != 1]   # annotation 개수가 그 이상인 것
random.shuffle(only1)

file_list = remain + only1

# 4. 정렬된 파일 순으로 라벨 개수 세기
for sub_list in file_list:
    '''
    sub_list = ['220708_BusanPort_00067_606.json', 6, {'fishing_plumb': 3, 'scrap_iron': 3}]

    sub_list[0] = '220708_BusanPort_00067_606.json'
    sub_list[1] = annotation 내 객체 총 개수
    sub_list[2] = annotation 내 클래스 별 객체 개수 모음 dictionary
    sub_list[2][클래스] = 클래스 별 객체 개수 

    [ 현재 세고 있는 라벨 개수 / 원하는 목표 라벨 개수 ] 
    labels_count[클래스][0] = 현재 세고 있는 라벨 개수
    labels_count[클래스][1] = 원하는 목표 라벨 개수
    '''
    sub_dict = sub_list[2]

    # 1) 먼저 full_list에 label 존재 여부 확인
    if not check_full_list(list(sub_dict.keys()), full_list):
        continue
    
    # 2) future_dict 생성
    future_dict = dict()
    for key in sub_dict.keys():
        future_dict[key] = list()
        future_dict[key].append(labels_count[key][0]+sub_dict[key])
        future_dict[key].append(check(labels_count[key][1], labels_count[key][0]+sub_dict[key]))

    # 3) 함수 check 리턴 값 확인
    if not is_two(future_dict):
        continue
    
    # 4) 목표치 채운 클래스 추가
    for key in future_dict.keys(): 
        if future_dict[key][1]==1:
            full_list.append(key)

    # 5) extracted_list에 추가 및 labels_count 업데이트
    extracted_list.append(sub_list[0])

    for key in future_dict.keys(): 
        labels_count[key][0] = future_dict[key][0]

# 5. extracted_list 토대로 img와 json을 분리해 저장
img_path = "/move/img/"     # 옮길 img 경로
json_path = "/move/json/"   # 옮길 json 경로

train_path = "/train/"      # train image 경로
test_path = "/test/"        # test image 경로
val_path = "/val/"          # val image 경로

file_path = "/total/"       # json 경로

train_list = os.listdir(train_path)
test_list = os.listdir(test_path)
val_list = os.listdir(val_path)

train_list = [ file_name for file_name in train_list if file_name.endswith(".png") ]
test_list = [ file_name for file_name in test_list if file_name.endswith(".png") ]
val_list = [ file_name for file_name in val_list if file_name.endswith(".png") ]

cnt = 0
for extract in extracted_list:
    cnt += 1
    print(f"{cnt}번: {extract}")
    path1 = file_path + extract
    path2 = json_path + extract
    os.system(f'cp {path1} {path2}')

    extract = extract[:-5]+'.png'
    if extract in train_list:
        path1 = train_path + extract
        path2 = img_path
        os.system(f'cp {path1} {path2}')
    elif extract in test_list:
        path1 = test_path + extract
        path2 = img_path
        os.system(f'cp {path1} {path2}')
    else:
        path1 = val_path + extract
        path2 = img_path
        os.system(f'cp {path1} {path2}')