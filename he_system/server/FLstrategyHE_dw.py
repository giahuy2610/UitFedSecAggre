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
from FLstrategyHE import FederatedMalwareStrategyHE

class FederatedMalwareStrategyHEDW(FederatedMalwareStrategyHE): 

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
        ########################### The dynamic weight update computation ######################################
        # loop through the results and update contribution (pairs of key, value) where
        # the key is the client id and the value is a dict of data size, sent size
        # and num_rounds_participated: updated value
        # total_data_size = 0
        for _,res in results:
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

        sumTemp = sum(self.dw_weight.values())
        for i in self.dw_weight:
            self.dw_weight[i] /= sumTemp
        
        ############## Weight Average using PyFHEl cyphertexts ##############
        client_results_in_cyphertext = []
        num_examples_total = sum([num_examples.num_examples for _, num_examples in results])
        
        for _, fit_res in results:

            one_client_cypher_results = np.array([])
            for enc_byte_obj in fit_res.parameters.tensors:   
                cypher_object = PyCtxt(pyfhel=HE, bytestring=enc_byte_obj)
                client_weight = fit_res.num_examples*self.dw_weight[fit_res.metrics["client_id"]]/num_examples_total
                one_client_cypher_results = np.append(one_client_cypher_results, cypher_object * client_weight)
            
            client_results_in_cyphertext.append(one_client_cypher_results)

        # Declare the variable to store the sum of the gradient
        cypher_sum_result = client_results_in_cyphertext[0]
        for k in range(1, len(client_results_in_cyphertext)):
            cypher_sum_result += client_results_in_cyphertext[k]

        avg_cypher_byte_list = []
        for end_cypher in cypher_sum_result:
            avg_cypher_byte_list.append(end_cypher.to_bytes())

        #####################################################################
        
        metrics_aggregated = {}
        return avg_cypher_byte_list, metrics_aggregated




