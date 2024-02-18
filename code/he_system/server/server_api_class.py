import json
from typing import Callable, Dict
import flwr as fl
from FLstrategyHE import FederatedMalwareStrategyHE
from FLstrategyHE_dw import FederatedMalwareStrategyHEDW

class ServerApi():
    def loadConfig(self):
        print("config json is importing ------")
        ##  Load config json
        with open('./config_training.json','r') as file:
            json_data = file.read()
        data = json.loads(json_data)
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
        self.he_enabled = data['he_enabled']
        
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
        # Create strategy
        strategy = FederatedMalwareStrategyHEDW(
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
                he_enabled=self.he_enabled
            )

        # Start Flower server
        fl.server.start_server(
            server_address= self.fl_server_address,
            config=fl.server.ServerConfig(num_rounds=self.fl_num_rounds),
            strategy=strategy,
        )

    def __init__(self) -> None:
        self.loadConfig()          


