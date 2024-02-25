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
from outlier_factor import cosine_similarity, cosine_similarity_normalization

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
        if server_round ==3:
            id=self.name
            if not os.path.exists(f"./results/{id}"):
                    os.makedirs(f"./results/{id}")
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
