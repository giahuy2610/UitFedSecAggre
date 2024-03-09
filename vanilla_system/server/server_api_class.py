import json
from typing import Callable, Dict
import flwr as fl
from strategy_avg import StrategyAvg 
from strategy_dwavg import StrategyDwAvg 
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
)
import tensorflow as tf
import os
import cv2
import numpy as np

class ServerApi():
    def load_img(self, datadir):
        img_arr = []
        target_arr = []
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

    def loadConfig(self):
        print("config json is importing ------")
        ##  Load config json
        with open('./config_training.json','r') as file:
            json_data = file.read()
        data = json.loads(json_data)
        self.data_categories = data["data_categories"]
        self.img_width = data["img_width"]
        self.img_height = data["img_height"]
        self.img_dim = data["img_dim"]
        self.fl_num_rounds = data['fl_num_rounds']
        self.fl_min_fit_clients = data['fl_min_fit_clients']     
        self.fl_min_evaluate_clients = data['fl_min_evaluate_clients']
        self.fl_min_available_clients = data['fl_min_available_clients']
        self.fl_aggregate_type = data['fl_aggregate_type']
        self.fl_server_address = data['fl_server_address']
        self.batch_size = data['batch_size']
        self.learning_rate = data['learning_rate']
        self.clt_local_epochs = data['clt_local_epochs']
        self.data_dir_path = data['data_dir_path']
        self.he_enabled = data['he_enabled']
        self.X_train, self.y_train= self.load_img(self.data_dir_path)
        print("config json is imported ------")


    def get_on_fit_config_fn(self) -> Callable[[int], Dict[str, str]]:
        """Return training configuration dict for each round.
        Keep batch size fixed at 32, 1 local epochs.
        """

        def fit_config(server_round: int) -> Dict[str, str]:       
            config = {
                "batch_size": self.batch_size,
                "local_epochs": self.clt_local_epochs,
                "learning_rate": self.learning_rate
            }
            return config
            
        return fit_config

    def launch_fl_session(self):
        """Start server and trigger update_strategy then connect to clients to perform fl session"""
        if (self.fl_aggregate_type == 0):
            # Create strategy
            strategy = StrategyAvg(
                    fraction_fit=1,
                    fraction_evaluate=1,
                    min_fit_clients=self.fl_min_fit_clients,
                    min_evaluate_clients=self.fl_min_evaluate_clients,
                    min_available_clients=self.fl_min_available_clients,
                    # evaluate_fn=get_evaluate_fn(model),
                    on_fit_config_fn=self.get_on_fit_config_fn(),
                    #on_evaluate_config_fn=evaluate_config,
                    #fit_metrics_aggregation_fn=weighted_average,
                    # evaluate_metrics_aggregation_fn=weighted_average,
                    fl_aggregate_type = self.fl_aggregate_type,
                    he_enabled=self.he_enabled, 
                    X_train=self.X_train,
                    y_train=self.y_train
            )
                   
        elif (self.fl_aggregate_type == 1):
            # Create strategy
            strategy = StrategyDwAvg(
                    fraction_fit=1,
                    fraction_evaluate=1,
                    min_fit_clients=self.fl_min_fit_clients,
                    min_evaluate_clients=self.fl_min_evaluate_clients,
                    min_available_clients=self.fl_min_available_clients,
                    # evaluate_fn=get_evaluate_fn(model),
                    on_fit_config_fn=self.get_on_fit_config_fn(),
                    #on_evaluate_config_fn=evaluate_config,
                    #fit_metrics_aggregation_fn=weighted_average,
                    # evaluate_metrics_aggregation_fn=weighted_average,
                    fl_aggregate_type = self.fl_aggregate_type,
                    he_enabled=self.he_enabled, 
                    X_train=self.X_train,
                    y_train=self.y_train
            )

        # Start Flower server
        fl.server.start_server(
            server_address= self.fl_server_address,
            config=fl.server.ServerConfig(num_rounds=self.fl_num_rounds),
            strategy=strategy,
        )

    def __init__(self) -> None:
        self.loadConfig()          


