'''
2023.05
xml 내용 기반으로 excel 파일을 만들기 위함
- Step1~Step4 유기적으로 이어짐
'''
import os
import pandas as pd
import zipfile
import xml.etree.ElementTree as ET

ROOT = '.\\EXTRACT\\EXTRACT'
PATH = os.path.join(ROOT, 'extract.zip')

'''
Step1. HumanID를 key값으로 하는 dict 생성
total_dict = {
'HumanID': [['upperclothes', 'lowerclothes', 'upperclothes_color', 'lowerclothes_color', 'filename'], ...]
}
'''
with zipfile.ZipFile(PATH, 'r') as zipObj:
    xml_files = zipObj.namelist()

    total_dict = dict()
    for xml in xml_files:
        with zipObj.open(xml) as xml:
            tree = ET.parse(xml)
            root = tree.getroot()
            obj = root.find("OBJECT")
            if not root.find("OBJECT"): continue

            temp_list = list()
            temp_list.append(obj.find('upperclothes').text)
            temp_list.append(obj.find('lowerclothes').text)
            temp_list.append(obj.find('upperclothes_color').text)
            temp_list.append(obj.find('lowerclothes_color').text)
            temp_list.append(root.find("FILE").find('name').text.split('.')[0]+'.xml')

            if obj.attrib['ID'] not in total_dict:    
                total_dict[obj.attrib['ID']] = [temp_list]
            else:
                total_dict[obj.attrib['ID']].append(temp_list)

'''
Step2. HumanID를 key값, 속성 4개 tuple을 key값으로 하는 dict를 value값으로 하는 dict 생성
result_dict = {
'HumanID': {('upperclothes', 'lowerclothes', 'upperclothes_color', 'lowerclothes_color'): ['filename', ...], ...}
}
'''
result_dict = dict()        
for k in total_dict.keys():
    sub_dict = dict()
    for sub in total_dict[k]:
        sub_tuple = tuple(sub[:4])
        if sub_tuple not in sub_dict:
            sub_dict[sub_tuple] = [sub[4]]
        else: 
            sub_dict[sub_tuple].append(sub[4])
    result_dict[k] = sub_dict
print('총 HumanID 개수: {}'.format(len(result_dict.keys())))

count = 0
for k in result_dict.keys():
    if len(result_dict[k].keys())>1:
        count += 1
print(f'속성이 다른 HumanID 개수: {count}')    

'''
Step3. DataFrame 생성 위한 작업
'''
h_id_list = []  #서로 다른 속성 그룹이 존재하는 h_id
par_list = []   #속성 그룹
fn_list = []    #파일명
for k in result_dict.keys():
    if len(result_dict[k].keys())==1: continue
    for idx in range(len(result_dict[k].keys())): #속성 그룹 개수만큼 h_id 칼럼 개수를 맞춰주기 위함
        h_id_list.append(k)

        sub_dict = result_dict[k]
        par_list.append(list(sub_dict.keys())[idx])
        fn_list.append(sub_dict[list(sub_dict.keys())[idx]])

'''
Step4. excel 변환
'''
validation = pd.DataFrame(
    {
        'HumanID': h_id_list,
        'Attribute(upperclothes, lowerclothes, upperclothes_color, lowerclothes_color)': par_list,
        'Filenames': fn_list,
})
validation.to_excel('./result_validation.xlsx', index=False)