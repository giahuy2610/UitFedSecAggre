import tensorflow as tf
import os
import cv2
import numpy as np
import json

def load_model_from_json():
    with open('model.json','r') as file:
        json_data = file.read()
    model = tf.keras.models.model_from_json(json_data)
    return model

def load_img_for_server(data_type):
    Categories = ['Locker', 'Mediyes', 'Winwebsec', 'Zbot', 'Zeroaccess']
    img_arr = []
    target_arr = []
    path=read_config_json_value('clt_test_data_path')
    datadir = path + data_type
    
    for i in Categories:
        print(f'loading... category : {i}')
        path = os.path.join(datadir, i)
        
        for img_file in os.listdir(path):
            # Đọc ảnh với OpenCV
            img = cv2.imread(os.path.join(path, img_file),cv2.IMREAD_GRAYSCALE)
            
            # Resize ảnh về kích thước 64x64
            img = cv2.resize(img, (64, 64))
            
            # Thêm ảnh vào mảng img_arr
            img_arr.append(img)
            
            # Thêm nhãn tương ứng vào mảng target_arr
            target_arr.append(Categories.index(i))
        
        print(f'loaded category: {i} successfully')
    
    # Chuyển đổi các mảng thành mảng NumPy
    img_arr = np.array(img_arr)
    target_arr = np.array(target_arr)
    
    return img_arr, target_arr

def read_config_json_value(key):
    with open('config_training.json','r') as file:
        json_data = file.read()
    data = json.loads(json_data)
    return data[key]

def read_api_key(service):
    with open('api_key.json','r') as file:
        json_data = file.read()
    data = json.loads(json_data)
    return data[service]["key"],data[service]["secret"]