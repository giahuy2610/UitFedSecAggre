import json
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import cv2
from typing import  Dict, List, Optional, Tuple, Union
from flwr.common.logger import log
from logging import WARNING
import flwr as fl
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import tensorflow as tf
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar
from flwr.common import NDArrays
from functools import reduce
from UitFedSecAggre.vanilla_system.Library.export_file_handler import save_weights
from UitFedSecAggre.vanilla_system.Library.reward_service import RewardService
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
reward_service = RewardService()

class StrategyAvg(fl.server.strategy.FedAvg): 
    #   FedAvg
    def aggregate(self, results: List[Tuple[NDArrays, int]],client_parameters)-> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = []
        for weights, num_examples in results:
            # Loop each client
            weighted_layer = []
            for layer in weights:
                # Loop each layer
                weighted_layer.append(layer * num_examples)
            weighted_weights.append(weighted_layer)        
        
        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        
        ''' REWARDING MECHANISM START '''
        amountPerRound = self.fl_budget
        self.model_architecture.set_weights(weights_prime)
        y_pred = self.model_architecture.predict(self.X_valid)
        y_pred_bool = np.argmax(y_pred, axis=1)
        #f1 score khi đầy đủ các client
        f1_score_orginal = f1_score(self.y_valid, y_pred_bool , average="macro",zero_division=0)

        # Compute contribution

        # List of f1-score for each leave-one-out case
        f1scoreLOO = []
        scaler=self.scaler
        
        number_of_round=self.max_round
        number_of_clients = len(results)
        #Lấy ra từng client trong results 
        client_id=[client_parameters[i][1].metrics['client_id'] for i in range(len(client_parameters))]

        # Loop to remove each client's local weight from global weight
        for i in range(len(results)):
            # Lưu weight không có sự đóng góp của client i
            results_loo = []

            for j in range(len(results)):
                if (i != j):
                    # Không sử dụng weight của client i
                    results_loo.append(results[i])

            #################BẮT ĐẦU FED AVG#############################
            num_examples_total_loo = sum([num_examples for _, num_examples in results_loo])

            # Create a list of weights, each multiplied by the related number of examples
            weighted_weights_loo = []

            for weights, num_examples in results_loo:
                # Loop each client
                weighted_layer = []
                for layer in weights:
                    # Loop each layer
                    weighted_layer.append(layer * num_examples)

                weighted_weights_loo.append(weighted_layer)

            # Compute average weights of each layer
            weights_prime_loo : NDArrays = [
                reduce(np.add, layer_updates) / num_examples_total_loo
                for layer_updates in zip(*weighted_weights_loo)
            ]

            self.model_architecture.set_weights(weights_prime_loo)
            y_pred = self.model_architecture.predict(self.X_valid)
            y_pred_bool = np.argmax(y_pred, axis=1)
            # Lưu kết quả testing
            f1scoreLOO.append(f1_score(self.y_valid, y_pred_bool , average="macro",zero_division=0))

        # Lưu chênh lệch giữa ko có và có sự đóng góp của client i
        f1scoreDeltaLOO = {}
        for i in range(len(results)):
            delta = f1_score_orginal - f1scoreLOO[i]
            idx=client_id[i]
            f1scoreDeltaLOO[idx]=delta
            """
                delta > 0 -> bỏ client i ra f1 score giảm -> client i có lợi
                delta < 0 -> bỏ client i ra mô hình tốt hơn -> client i có hại
            """
        #sort f1scoreDeltaLOO theo value
        f1scoreDeltaLOO = dict(sorted(f1scoreDeltaLOO.items(), key=lambda item: item[1]))
        #Minmax scaling các value trong f1scoreDeltaLOO
        arr=scaler.fit_transform(np.array(list(f1scoreDeltaLOO.values())).reshape(-1,1)).reshape(-1)
        #Gán lại value cho f1scoreDeltaLOO
        f1scoreDeltaLOO={list(f1scoreDeltaLOO.keys())[i]:arr[i] for i in range(len(arr))}
        """ Nếu toàn bộ các client khi bỏ ra đều làm mô hình tốt lên (delta âm) -> mô hình đang overfitting -> chỉ giữ lại 1 nửa weight )"""
        if len([s for s in f1scoreDeltaLOO.values() if s < 0]) >= number_of_clients/2:
            # Trung vị
            median = np.median(list(f1scoreDeltaLOO))
            for i in client_id:
                # Nếu f1score của client i > median thì lấy tuyệt đối (để chút tính tiền chứ hiện tại đang âm)
                # Ngược lại thì gán bằng 0 -> ko đc trả payoff
                f1scoreDeltaLOO[i] = abs(f1scoreDeltaLOO[i]) if f1scoreDeltaLOO[i] > median else 0
        
        #sort f1scoreDeltaLOO theo key 
        f1scoreDeltaLOO = dict(sorted(f1scoreDeltaLOO.items(), key=lambda item: item[0]))
        # tính tổng delta
        sumF1scoreDelta = sum(list(f1scoreDeltaLOO.values()))
        # Proof of work (đồng đều, ai cũng được tiền)
        reward_scores=amountPerRound * 0.2 / (number_of_clients* number_of_round)
        payoffByClient = {}
        for element in f1scoreDeltaLOO.keys():

            # Proof of performance (trả theo đóng góp - dựa trên delta) 
            reward_scores_temp =   (f1scoreDeltaLOO[element] / sumF1scoreDelta) * ( amountPerRound * 0.8/number_of_round)  

            # Payoff of each client
            payoffByClient[element] = int(math.floor(reward_scores_temp + reward_scores)) 
            ''' REWARDING MECHANISM END '''
        
        self.weight_history.append(weights_prime[-1])
        print(f"payoffByClient reward: {payoffByClient}")
        #Ghi lại số token cho client
        for key in payoffByClient.keys():
            if key in self.payoffByClient:
                self.payoffByClient[key]+=payoffByClient[key]
            else:
                self.payoffByClient[key]=payoffByClient[key]
        print(f"payoffByClient total: {self.payoffByClient}")
        return weights_prime

    def __init__(self,
        *,
        #optional parameters for customizations
        #name: Optional[str] = None,
        fraction_fit=0.7, # Use 70% samples of available clients for training
        fraction_evaluate=0.2, # Use 20% samples of available clients for evaluation
        min_fit_clients=2, # At least 1 client is needed for training
        min_evaluate_clients=2, # At least 1 client is needed for evaluation
        min_available_clients=2, # Wait until all 1 clients are available
        evaluate_fn=None,
        on_fit_config_fn=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        fl_aggregate_type = 0,
        he_enabled=True,
        ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            fit_metrics_aggregation_fn = fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.contribution={
            'total_data_size': 0
        }
        self.result={
            'aggregated_loss':{
                'server':{},
                'client':{}
            },
            'aggregated_accuracy':{
                # 0:0
                'server':{},
                'client':{}
            },
            'f1_score':{}
        }
        self.dw_weight = {}
        self.dw_accp = {}
        self.factor=0.2
        self.fl_aggregate_type = fl_aggregate_type
        self.weight_history = []
        self.current_server_round = 0
        self.he_enabled = he_enabled
        self.scaler = MinMaxScaler()

        if self.he_enabled:
            print('running with HE')
        else:
            print("running without HE")
        self.session=None
        with open('config_training.json') as f:
            data=json.load(f)
            self.max_round = data['fl_num_rounds']
            self.session=data['session']

            self.img_width = data["img_width"]
            self.img_height = data["img_height"]
            self.img_dim = data["img_dim"]
            self.data_dir_path = data['data_dir_path']
            self.data_categories = data['data_categories']
            self.fl_budget=data['fl_budget']

            self.df_optimizer_type = data["df_optimizer_type"]
            self.l2_norm_clip = data['df_l2_norm_clip']
            self.noise_multiplier = data['df_noise_multiplier']
            self.num_microbatches = data['df_num_microbatches']
            self.isMalwareDetection = data['malware_detection']
            self.model=self.generate_cnn_model()
            self.X_valid, self.y_valid = self.load_img('valid', self.data_dir_path)
            self.current_round_weight=None
            if self.model is None:
                raise ValueError("Data is not set. Please set the model before calling aggregate_evaluate.")
            if self.X_valid is None:
                raise ValueError("Data is not set. Please set the data before calling aggregate_evaluate.")
        self.payoffByClient = {}

    def custom_aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Store current round number
        self.current_server_round = server_round

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        parameters_aggregated = ndarrays_to_parameters(self.aggregate(weights_results,results))
        

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_weights = self.custom_aggregate_fit(server_round, results, failures)

        save_weights(aggregated_weights, self.session, server_round)
        # for i in range(len(results)):
        #     result = results[i]
        #     payoff = self.payoffByClient[i]
        #     wallet_address=result[1].metrics['wallet_address']
        #     print(f"payoff for client {result[1].metrics['client_id']} is {payoff}")
        #     reward_service.pay(wallet_address, payoff)
        if server_round == self.max_round:
            for i in range(len(results)):
                result=results[i]
                client_id = result[1].metrics['client_id']
                payoff = self.payoffByClient[client_id]
                wallet_address=result[1].metrics['wallet_address']
                print(f"payoff for client {client_id} is {payoff}")
                reward_service.pay(wallet_address, payoff)

        self.current_round_weight = parameters_to_ndarrays(aggregated_weights[0])
        return aggregated_weights

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        """Aggregate evaluation accuracy using weighted average."""
        if not results:
            return None, {}
        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        #server tự evaluate
        self.model.set_weights(self.current_round_weight)
        aggregated_loss, aggregated_accuracy = self.model.evaluate(self.X_valid, self.y_valid)
        self.result['aggregated_loss']['server'][server_round]=aggregated_loss
        self.result['aggregated_accuracy']['server'][server_round]=aggregated_accuracy

        #eval theo client
        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        self.result['aggregated_loss']['client'][server_round]=aggregated_loss
        self.result['aggregated_accuracy']['client'][server_round]=aggregated_accuracy

        y_pred = self.model.predict(self.X_valid)
        y_pred_bool = np.argmax(y_pred, axis=1)
        self.result['f1_score'][server_round]=f1_score(self.y_valid, y_pred_bool , average="macro",zero_division=0)

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}
    
    def generate_cnn_model(self):
        print("cnn model is creating -----")
        if self.isMalwareDetection:
            model_file = 'model_detection.json'
        else:
            model_file = 'model.json'
        with open(model_file,'r') as file:
            json_data = file.read()
        self.model_architecture = tf.keras.models.model_from_json(json_data)
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
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
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
            #Kiểm tra xem thư mục có tồn tại không
            if os.path.isdir(path):
                for img_file in os.listdir(path):
                    # Đọc ảnh với OpenCV
                    img = cv2.imread(os.path.join(path, img_file),cv2.IMREAD_GRAYSCALE)
                    
                    # Resize ảnh về kích thước 64x64
                    img = cv2.resize(img, (int(self.img_width), int(self.img_height)))
                    if self.isMalwareDetection == False:
                        if i != "benign":
                            # Thêm ảnh vào mảng img_arr
                            img_arr.append(img)
                            target_arr.append(Categories.index(i))
                    else:
                        # Thêm ảnh vào mảng img_arr
                        img_arr.append(img)
                        target_arr.append(0 if i == "benign" else 1) 
                
                print(f'loaded category: {i} successfully')
        
        # Chuyển đổi các mảng thành mảng NumPy
        img_arr = np.array(img_arr)
        target_arr = np.array(target_arr)
        return img_arr, target_arr