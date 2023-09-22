import csv

def read_csv(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f) #컬럼 내용 넘기기
        next(csv_reader)
        for line in csv_reader: 
            fn_name, PID, CAMID = line