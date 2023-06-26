import os
import csv
#%%
ROOT = '/home/'
GALLERY_TEST_PATH = os.path.join(ROOT, 'gallery/bounding_box_test') #gallery 이미지 폴더 -> csv 내 fn_name과 연결할 것(GALLERY_TEST_PATH와 csv 내 fn_name과 하나하나 join)
IMG_GALLERY_TEST_PATH = os.path.join(GALLERY_TEST_PATH, 'img')
CSV_GALLERY_TEST_PATH = os.path.join(GALLERY_TEST_PATH, 'csv')
QUERY_PATH = os.path.join(ROOT, 'query') #query 이미지 폴더 -> csv명과 연결할 것(QUERY_PATH와 csv명과 join)
REPOSITORY = os.path.join(ROOT, 'htmls') #html 파일이 저장될 곳 -> csv명과 동일하게 파일 제작

query_images = os.listdir(QUERY_PATH)
print(query_images)
#%%
def head():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Re-ID Project</title>
        <link rel="stylesheet" href="../style.css">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=PT+Serif&display=swap" rel="stylesheet">
    </head>
    """

def body_header():
    return """
    <header class="title-container">
        <div>
            <span class="status"></span>
            <h1 class="page-title">결과목록</h1>
        </div>
    </header>
    """

def query_container(query_path, query_name):
    query_container_header =  """
    <header class="query-container-header">
        <div>
            <span class="status"></span>
            <h1 class="query-title">Query Image</h1>
        </div>
    </header>
    """

    query_person = f'<ul class="query-list">\
                        <li class="query-person">\
                            <img src="{query_path}" alt=""> \
                            <div class="id-info">\
                                <span class="ped">보행자 ID</span>\
                                <span class="ped-id">{query_name}</span>\
                            </div>\
                        </li>\
                    </ul>\
                    '

    return '<aside class="query-container">'+query_container_header+query_person+'</aside>'

def cctv_container_header(CAMID):
    return (
        f'<header class="cctv-container-title">\
            <h2 class="cctv-id">{CAMID}</h2>\
            <div class="cctv-content">\
                <span>2023-04-25</span>\
                <span>09:00:27~09:19:32</span>\
                <span>00:19:28</span>\
                <span>■ 0/10</span>\
            </div>\
        </header>'\
    )

def make_gallery_person(gallery_path, PID):
    return (
        f'<li class="gallery-person">\
            <img src="{gallery_path}" alt="">\
            <div class="id-info">\
                <span class="ped">보행자</span>\
                <span class="ped-id">{PID}</span>\
            </div>\
        </li>'\
    )   

def gallery_container_header():
    return """
    <header class="gallery-container-header">
        <div>
            <span class="status"></span>
            <h1 class="gallery-title">Gallery</h1>
        </div>
    </header>
    """

def make_cctv_container(CAMID, gallery_persons):
    cctv_container = '<article class="cctv-container">'+cctv_container_header(CAMID)
    gallery_string = '<ul class="gallery-list">'
    gallery_ = list()

    for person in gallery_persons:
        gallery_path = os.path.join(IMG_GALLERY_TEST_PATH, person[0])
        gallery_.append(make_gallery_person(gallery_path,  person[0].split('.')[0]))
    gallery_ = ''.join(gallery_)

    return cctv_container+gallery_string+gallery_+'</ul></article>'

def make_html_text(query_image, csv_name, html_name):
    query_image_path = os.path.join(QUERY_PATH, query_image)
    csv_path = os.path.join(CSV_GALLERY_TEST_PATH, csv_name)

    cctv_dict = dict()

    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for line in csv_reader:
            fn_name, PID, CAMID = line #gallery 파일명, PersonID, CCTVID
            if CAMID not in cctv_dict:
                cctv_dict[CAMID] = list()
                cctv_dict[CAMID].append((fn_name, PID))
            else: 
                cctv_dict[CAMID].append((fn_name, PID))

        query_container_ = query_container(query_image_path, query_image.split('.')[0])
        html_str = head()+'<body>'+body_header()+'<main class="main-container">'+query_container_
        
        cctv_ = list()
        for CAMID, gallery_persons in cctv_dict.items():
            cctv_.append(make_cctv_container(CAMID, gallery_persons))
        cctv_ = ''.join(cctv_)
        gallery_container = '<section class="gallery-container">'+gallery_container_header()+'<div>'
        gallery_container+=cctv_+'</div>'+'</section>'

        html_str+=gallery_container+'</main>'+'</body>'+'</html>'
        #%%
        html_name = os.path.join(REPOSITORY, html_name)
        html_file = open(html_name, 'w')
        html_file.write(html_str)
        html_file.close()
        print(fn_name, PID, CAMID)
#%%
if __name__ == "__main__":
    #%%
    for query_image in query_images:
        #%%
        try:
            html_name = query_image.split('.')[0]+'.html'
            csv_name = query_image.split('.')[0]+'.csv'
            make_html_text(query_image, csv_name, html_name)
        except:
            pass

