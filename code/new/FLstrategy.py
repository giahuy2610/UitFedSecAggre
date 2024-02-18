import os
import numpy as np
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
from Pyfhel import Pyfhel, PyPtxt, PyCtxt


def cosine_similarity(vector1, vector2):
    # Chuyển đổi vector thành các tensor trong TensorFlow
    v1 = tf.convert_to_tensor(vector1, dtype=tf.float32)
    v2 = tf.convert_to_tensor(vector2, dtype=tf.float32)

    # Reshape vector để có shape (batch_size, 1)
    v1 = tf.reshape(v1, (1, -1))
    v2 = tf.reshape(v2, (1, -1))

    # Tính cosine similarity
    similarity_loss = tf.keras.losses.CosineSimilarity(axis=1)(v1, v2)

    return abs(similarity_loss.numpy())


def cosine_similarity_normalization(client_distance_array):
    new_client_distance_array = []
    for client_distance in client_distance_array:
        new_client_distance_array.append((client_distance-min(client_distance_array))/(max(client_distance_array)-min(client_distance_array)))
    
    return new_client_distance_array


def outlier_factor():
    return

class FederatedMalwareStrategy(fl.server.strategy.FedAvg): 
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

    #   DW-FedAvg
    def aggregateCustomDWFedAvg(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        multiplied_results = []
        for i, (weights, num_examples) in enumerate(results):
            multiplied_num_examples = num_examples * self.dw_weight[i]
            multiplied_results.append((weights, multiplied_num_examples))
        results =  multiplied_results

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]
        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
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
        he_enabled=True
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
        self.name='noise_0.1'
        self.contribution={
            'total_data_size': 0
        }
        self.result={
            'aggregated_loss':{
                0:0
            },
            'aggregated_accuracy':{
                0:0
            }
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
        # log("sos custom aggregate fit")
        # log(weights_results)
        if (self.fl_aggregate_type == 0):
            parameters_aggregated = ndarrays_to_parameters(self.aggregate(weights_results))

        elif (self.fl_aggregate_type == 1):
            parameters_aggregated = ndarrays_to_parameters(self.aggregateCustomDWFedAvg(weights_results))

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
            clientId = res.metrics["client_id"]
            
            # results: List[Tuple[ClientProxy, FitRes]]
            # FitRes: parameters: Parameters , num_examples: int , metrics: Optional[Metrics] = None
            if server_round == 1:
                self.dw_accp[clientId] = res.metrics["accuracy"]
                self.dw_weight[clientId] = 1/len(results)
            else:    
                if res.metrics["accuracy"] > self.dw_accp[clientId]:
                    self.dw_weight[clientId] *= 1+self.factor
                elif res.metrics["accuracy"] > self.dw_accp[clientId]:
                    self.dw_weight[clientId] *= 1-self.factor
            
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
        if server_round ==3:
            id=self.name
            if not os.path.exists(f"./result/{id}"):
                    os.makedirs(f"./result/{id}")
            np.save(f"./result/{id}/{id}_model_weights.npy", aggregated_weights)


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

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        self.result['aggregated_loss'][server_round]=aggregated_loss
        self.result['aggregated_accuracy'][server_round]=aggregated_accuracy

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}


# def get_evaluate_fn(model):
#     """Return an evaluation function for server-side evaluation."""
#     # Load data and model here to avoid the overhead of doing it in `evaluate` itself
#     X_valid,y_valid=load_img('valid')
#     X_valid = X_valid/255

#     # The `evaluate` function will be called after every round
#     def evaluate(
#         server_round: int,
#         parameters: fl.common.NDArrays,
#         config: Dict[str, fl.common.Scalar],
#     ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
#         model.set_weights(parameters)  # Update model with the latest parameters
#         loss, accuracy = model.evaluate(X_valid, y_valid)
#         return loss, {"accuracy": accuracy}
#     return evaluate


# def evaluate_config(server_round: int):
#     """Return evaluation configuration dict for each round.
#     Perform five local evaluation steps on each client (i.e., use five
#     batches) during rounds one to three, then increase to ten local
#     evaluation steps.
#     """
#     val_steps = 5 
#     return {"val_steps": val_steps, "round": server_round }


# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
    
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}


# def load_model():
#     with open('/Users/vfa/Desktop/huytg/khoaluan/malwareClassification/code/new/model_architecture.json','r') as file:
#         json_data = file.read()
#     model = tf.keras.models.model_from_json(json_data)
#     noise_multiplier = 0.1
#     optimizer = DPKerasAdamOptimizer(
#         l2_norm_clip=1.0,
#         noise_multiplier=noise_multiplier,
#         num_microbatches=1,
#         )
#     model.compile(optimizer, tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
#     return model

# Categories = ['Locker', 'Mediyes', 'Winwebsec', 'Zbot', 'Zeroaccess']
# def load_img(data_type):
#     img_arr = []
#     target_arr = []
#     datadir = '/Users/vfa/Desktop/huytg/khoaluan/malwareClassification/code/classedExeImg/' + data_type
    
#     for i in Categories:
#         print(f'loading... category : {i}')
#         path = os.path.join(datadir, i)
        
#         for img_file in os.listdir(path):
#             # Đọc ảnh với OpenCV
#             img = cv2.imread(os.path.join(path, img_file))
            
#             # Resize ảnh về kích thước 64x64
#             img = cv2.resize(img, (64, 64))
            
#             # Thêm ảnh vào mảng img_arr
#             img_arr.append(img)
            
#             # Thêm nhãn tương ứng vào mảng target_arr
#             target_arr.append(Categories.index(i))
        
#         print(f'loaded category: {i} successfully')
    
#     # Chuyển đổi các mảng thành mảng NumPy
#     img_arr = np.array(img_arr)
#     target_arr = np.array(target_arr)
    
#     return img_arr, target_arr


