import os 
import json

#클래스별 annotation 개수 담을 dictionary
labels = {
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

file_path = "/total"
file_list = os.listdir(file_path)

for file_name in file_list:
    if not file_name.endswith(".json"): #해당 파일이 json 파일이 아니면 넘어가기
        continue
    json_path = file_path + "/" + file_name
    with open(json_path, 'r') as file:  
        data = json.load(file)
        for d_shape in data["shapes"]:
            labels[d_shape["label"]] += 1  #클래스별 annotation 확인
    
print(labels)