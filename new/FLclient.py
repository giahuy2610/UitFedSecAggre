import flwr as fl
import json

class Malwarelient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_id) -> None:
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.client_id = client_id
        self.model = model
        self.client_address = ''


    def get_parameters(self,config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        learning_rate: int = config["learning_rate"]
        # round: int = config["round"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size,
            epochs
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.X_train)
        results = {
            "client_id": self.client_id,
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "client_address": self.client_address,            
        }
        # address = ''
        # with open('client_setting/client'+ self.client_id +'.json') as f:
        #     data = json.load(f)
        #     address=data['address']
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": float(accuracy)}
    
