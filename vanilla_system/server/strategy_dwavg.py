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

from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar
from flwr.common import NDArrays
from functools import reduce
from strategy_avg import StrategyAvg


class StrategyDwAvg(StrategyAvg): 
    #   DW-FedAvg
    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        multiplied_results = []
        print(self.dw_weight)
        for i, (weights, num_examples) in enumerate(results):
            multiplied_num_examples = num_examples * self.dw_weight[self.clientId]
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
        # results: List[Tuple[ClientProxy, FitRes]]
        # FitRes: parameters: Parameters , num_examples: int , metrics: Optional[Metrics] = None
        return weights_prime
