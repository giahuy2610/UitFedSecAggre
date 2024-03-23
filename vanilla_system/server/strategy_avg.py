import json
import numpy as np
from sklearn.metrics import f1_score
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
from outlier_factor import cosine_similarity, cosine_similarity_normalization
from UitFedSecAggre.vanilla_system.Library.reward_service import RewardService
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
reward_service = RewardService()

class StrategyAvg(fl.server.strategy.FedAvg): 
    #   FedAvg
    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = []
        client_distance_array = []
        for weights, num_examples in results:
            # Loop each client
            weighted_layer = []
            for layer in weights:
                # Loop each layer
                weighted_layer.append(layer * num_examples)
            if (self.current_server_round > 1):
                client_distance_array.append(cosine_similarity(weights[-1],self.weight_history[self.current_server_round - 2]))
            weighted_weights.append(weighted_layer)
    
        print(client_distance_array)
        client_distance_array = cosine_similarity_normalization(client_distance_array)
        self.threshold = np.average(client_distance_array)
        
        
        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        

        self.weight_history.append(weights_prime[-1])
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

            self.df_optimizer_type = data["df_optimizer_type"]
            self.l2_norm_clip = data['df_l2_norm_clip']
            self.noise_multiplier = data['df_noise_multiplier']
            self.num_microbatches = data['df_num_microbatches']
            self.model=self.generate_cnn_model()
            self.X_valid, self.y_valid = self.load_img('valid', self.data_dir_path)
            self.current_round_weight=None
            if self.model is None:
                raise ValueError("Data is not set. Please set the model before calling aggregate_evaluate.")
            if self.X_valid is None:
                raise ValueError("Data is not set. Please set the data before calling aggregate_evaluate.")

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

        parameters_aggregated = ndarrays_to_parameters(self.aggregate(weights_results))

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
        
        # loop through the results and update contribution (pairs of key, value) where
        # the key is the client id and the value is a dict of data size, sent size
        # and num_rounds_participated: updated value
        # total_data_size = 0
        for resTuple in results:
            res = resTuple[1]
            self.clientId = res.metrics["client_id"]
            
            # results: List[Tuple[ClientProxy, FitRes]]
            # FitRes: parameters: Parameters , num_examples: int , metrics: Optional[Metrics] = None
            if server_round == 1:
                self.dw_accp[self.clientId] = res.metrics["accuracy"]
                self.dw_weight[self.clientId] = 1/len(results)
            else:    
                if res.metrics["accuracy"] > self.dw_accp[self.clientId]:
                    self.dw_weight[self.clientId] *= 1+self.factor
                elif res.metrics["accuracy"] > self.dw_accp[self.clientId]:
                    self.dw_weight[self.clientId] *= 1-self.factor
            
            if res.metrics['client_id'] not in self.contribution.keys():
                self.contribution[res.metrics["client_id"]]={
                    "data_size":res.num_examples,
                    "num_rounds_participated":1,
                    "client_address":res.metrics['client_address']
                }
                self.contribution['total_data_size'] = self.contribution['total_data_size']+res.num_examples
            else:
                self.contribution[res.metrics["client_id"]]["num_rounds_participated"]+=1

            # print("data size = ", res.num_examples)
            # print("client id = ",clientId)
            # print("client weight = ",self.dw_weight[clientId])

        sumTemp = sum(self.dw_weight.values())
        for i in self.dw_weight:
            self.dw_weight[i] /= sumTemp
            # print("client id = ", i)
            # print("client weight after = ",self.dw_weight[i])


        aggregated_weights = self.custom_aggregate_fit(server_round, results, failures)
        
        save_weights(aggregated_weights, self.session, server_round)
        if server_round == self.max_round:
            for result in results:
                wallet_address=result[1].metrics['wallet_address']
                reward_service.payEveryoneEqually(wallet_address, 10)
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
        with open('model.json','r') as file:
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