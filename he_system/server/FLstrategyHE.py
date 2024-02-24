import numpy as np
from typing import  Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)

from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import Scalar
from Pyfhel import Pyfhel, PyCtxt
import datetime

class FederatedMalwareStrategyHE(fl.server.strategy.FedAvg):     
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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print("FederatedMalwareStrategyHE aggreate_fit",datetime.datetime.now())
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Store current round number
        self.current_server_round = server_round

        HE = Pyfhel() # Empty creation
        HE.load_context("./context")
        
        ############## Weight Average using PyFHEl cyphertexts ##############
        client_results_in_cyphertext = []
        for _, fit_res in results:

            one_client_cypher_results = np.array([])
            for enc_byte_obj in fit_res.parameters.tensors:   
                cypher_object = PyCtxt(pyfhel=HE, bytestring=enc_byte_obj)
                one_client_cypher_results = np.append(one_client_cypher_results, cypher_object*fit_res.num_examples)

            client_results_in_cyphertext.append(one_client_cypher_results)

        # Declare the variable to store the sum of the gradient
        cypher_sum_result = client_results_in_cyphertext[0]
        for k in range(1, len(client_results_in_cyphertext)):
            cypher_sum_result += client_results_in_cyphertext[k]

        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples.num_examples for _, num_examples in results])
        cypher_sum_result *= np.float32(1/num_examples_total)

        avg_cypher_byte_list = []
        for end_cypher in cypher_sum_result:
            avg_cypher_byte_list.append(end_cypher.to_bytes())

        #####################################################################
        
        metrics_aggregated = {}
        return avg_cypher_byte_list, metrics_aggregated
        
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager # Parameters
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        print("FederatedMalwareStrategy configure_evaluate",datetime.datetime.now())
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        new_params = Parameters(tensors=parameters, tensor_type="numpy.ndarray")

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        
        evaluate_ins = EvaluateIns(new_params, config) # parameters

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager # Parameters
    ) -> List[Tuple[ClientProxy, FitIns]]:
        print("FederatedMalwareStrategy configure_fit",datetime.datetime.now())
        """Configure the next round of training."""

        new_params = None

        if type(parameters) is type([]):
            new_params = Parameters(tensors=parameters, tensor_type="numpy.ndarray")
        else:
            new_params = parameters

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(new_params, config) # parameters

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
     
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        print("FederatedMalwareStrategy evaluate",datetime.datetime.now())
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)

        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics     




