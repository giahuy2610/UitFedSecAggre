import flwr as fl
import string
import random
from FLstrategy import FederatedMalwareStrategy
from server_api import ClientApi

def client_fn(cid: str):
    # Return a standard Flower client
    def get_random_string(length):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str
    return ClientApi().getInstance(get_random_string(8))

def launch_fl_simulator(self):
    strategy = FederatedMalwareStrategy(
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
            fl_aggregate_type = self.fl_aggregate_type
        )
    # Launch the simulation
    hist = fl.simulation.start_simulation(
    client_fn=client_fn, # A function to run a _virtual_ client when required
    num_clients=4, # Total number of clients available
    config=fl.server.ServerConfig(num_rounds=3), # Specify number of FL rounds
    strategy=strategy # A Flower strategy
)

# Launch the simulation
hist = launch_fl_simulator()