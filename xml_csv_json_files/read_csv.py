'''
2023.05
csv 파일을 read 하기 위함
'''
import csv

def read_csv(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for line in csv_reader: 
            fn_name, PID, CAMID = line