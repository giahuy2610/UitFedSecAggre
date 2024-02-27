import json
from pathlib import Path
import string
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
import flwr as fl
import cv2
import os
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from client import Client

class ClientApi():
    def loadConfig(self):
        print("config json is importing ------")
        ##  Load config json
        with open('config_training.json','r') as file:
            json_data = file.read()
        data = json.loads(json_data)
        self.data_categories = data["data_categories"]
        self.img_width = data["img_width"]
        self.img_height = data["img_height"]
        self.img_dim = data["img_dim"]
        self.imbalanced_type = data["imbalanced_type"]
        self.l2_norm_clip = data['df_l2_norm_clip']
        self.noise_multiplier = data['df_noise_multiplier']
        self.num_microbatches = data['df_num_microbatches']
        self.df_optimizer_type = data["df_optimizer_type"]
        self.fl_num_rounds = data['fl_num_rounds']
        self.fl_min_fit_clients = data['fl_min_fit_clients']
        self.fl_min_evaluate_clients = data['fl_min_evaluate_clients']
        self.fl_min_available_clients = data['fl_min_available_clients']
        self.fl_aggregate_type = data['fl_aggregate_type']
        self.fl_server_address = data['fl_server_address']
        self.batch_size = data['batch_size']
        self.learning_rate = data['learning_rate']
        self.clt_local_epochs = data['clt_local_epochs']
        self.clt_data_path = data['clt_data_path']

        self.session= data['session']
        self.wallet_address=None
        print("config json is imported ------")


    def dataImblanced(self, X_train, y_train):

        # Define balanced data generator
        def getSMOTEData(X, y, random = 42):
            smote = SMOTE(random_state= random)
            return smote.fit_resample(X, y)


        def getOSData(X, y, random = 42):
            os = RandomOverSampler(random_state= random)
            return os.fit_resample(X, y)


        def getUSData(X, y, random = 42):
            us = RandomUnderSampler(random_state= random)
            return us.fit_resample(X, y)
        
       
        batch_size, width, height  = X_train.shape
        
        print(self.imbalanced_type)

        match self.imbalanced_type:
            case 0:
                print("sos")
                return X_train, y_train
            case 1:
                 return getSMOTEData(X_train.reshape(batch_size, width * height), y_train)
            case 2:
                return getOSData(X_train.reshape(batch_size, width * height), y_train)
            case 3:
                return getUSData(X_train.reshape(batch_size, width * height), y_train)

    def loadModel(self):
        print("model json is importing ------")
        ##  Load model json``
        with open('model.json','r') as file:
            json_data = file.read()
        self.model_architecture = tf.keras.models.model_from_json(json_data)
        print("model json is imported ------")

    def generate_cnn_model(self):
        print("cnn model is creating -----")
        match self.df_optimizer_type :
            case 0:
                optimizer = "adam"
            case 1:
                optimizer=dp_optimizer_keras.DPKerasAdamOptimizer(
                l2_norm_clip= float(self.l2_norm_clip),
                noise_multiplier= float(self.noise_multiplier),
                num_microbatches= self.num_microbatches)      
            case 2:
                optimizer=dp_optimizer_keras.DPKerasSGDOptimizer(
                l2_norm_clip= float(self.l2_norm_clip),
                noise_multiplier= float(self.noise_multiplier),
                num_microbatches= self.num_microbatches)      
            case 3:
                optimizer=dp_optimizer_keras.DPKerasAdagradOptimizer(
                l2_norm_clip= float(self.l2_norm_clip),
                noise_multiplier= float(self.noise_multiplier),
                num_microbatches= self.num_microbatches) 

        self.model_architecture.compile(optimizer=optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        print("cnn model is created ------")
        return self.model_architecture

    def load_img(self, data_type, datadir):
        img_arr = []
        target_arr = []
        datadir = datadir + data_type
        Categories = self.data_categories
        
        for i in Categories:
            print(f'loading... category : {i}')
            path = os.path.join(datadir, i)
            
            for img_file in os.listdir(path):
                # Đọc ảnh với OpenCV
                img = cv2.imread(os.path.join(path, img_file),cv2.IMREAD_GRAYSCALE)
                
                # Resize ảnh về kích thước 64x64
                img = cv2.resize(img, (int(self.img_width), int(self.img_height)))
                
                # Thêm ảnh vào mảng img_arr
                img_arr.append(img)
                
                # Thêm nhãn tương ứng vào mảng target_arr
                target_arr.append(Categories.index(i))
            
            print(f'loaded category: {i} successfully')
        
        # Chuyển đổi các mảng thành mảng NumPy
        img_arr = np.array(img_arr)
        target_arr = np.array(target_arr)
        return img_arr, target_arr

    def launch_fl_session(self, client_id: string):
        data_path=self.clt_data_path+'client'+client_id+'/'
        X_train,y_train= self.load_img('train', data_path)

        X_train,y_train=self.dataImblanced( X_train,y_train)
        X_train = X_train.reshape(X_train.shape[0], self.img_width, self.img_height)

        X_test,y_test=self.load_img('test', data_path)
        # X_train = X_train/255
        # X_test = X_test/255

        with open('client'+client_id+'.json','r') as file:
            self.wallet_address = json.load(file)["address"] 
        fl.client.start_numpy_client(
            server_address=self.fl_server_address, 
            client=Client(self.model_architecture  ,
                          X_train, y_train, 
                          X_test, y_test, client_id, 
                          self.session,self.wallet_address),
            root_certificates=Path("../.cache/certificates/ca.crt").read_bytes(),
        )

    def __init__(self) -> None:
        self.loadModel()
        self.loadConfig()
        self.generate_cnn_model()



