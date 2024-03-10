from datetime import datetime
import json
from pathlib import Path
import time
from typing import Callable, Dict
import flwr as fl
from strategy_avg import StrategyAvg 
from strategy_dwavg import StrategyDwAvg 
from UitFedSecAggre.vanilla_system.Library.export_file_handler import save_config_file, write_json_result_for_server

class ServerApi():
    def loadConfig(self):
        print("config json is importing ------")
        ##  Load config json
        with open('./config_training.json','r') as file:
            json_data = file.read()
        data = json.loads(json_data)
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
        
        
        self.session=data['session']
        print("config json is imported ------")

    def get_on_fit_config_fn(self) -> Callable[[int], Dict[str, str]]:
        """Return training configuration dict for each round.
        Keep batch size fixed at 32, 1 local epochs.
        """

        def fit_config(server_round: int) -> Dict[str, str]:       
            config = {
                "batch_size": self.batch_size,
                "local_epochs": self.clt_local_epochs,
                "learning_rate": self.learning_rate,
                "round": server_round,
            }
            return config
            
        return fit_config

    def launch_fl_session(self):
        """Start server and trigger update_strategy then connect to clients to perform fl session"""
        start_time =time.time()
        date_time = datetime.now()
        current_time = date_time.strftime("%H:%M:%S")
        date= date_time.strftime("%d/%m/%Y")
        
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
                    he_enabled=self.he_enabled
            )

        # Start Flower server
        fl.server.start_server(
            server_address= self.fl_server_address,
            config=fl.server.ServerConfig(num_rounds=self.fl_num_rounds),
            strategy=strategy,
            certificates=(
                Path("../.cache/certificates/ca.crt").read_bytes(),
                Path("../.cache/certificates/server.pem").read_bytes(),
                Path("../.cache/certificates/server.key").read_bytes(),
            )
        )
        end_time = time.time()
        total_time = end_time - start_time

        dictionary = {
            "date": date,
            "start_time": current_time,
            "total_time": total_time,
        }
        
        write_json_result_for_server(strategy.result, self.session)
        save_config_file('config_training.json', self.session, dictionary)

    

    def __init__(self) -> None:
        self.loadConfig()          