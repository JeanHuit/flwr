import flwr as fl
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Define the server class
class CifarServer:
    def __init__(self):
        # Initialize global model weights (you should define your model here)
        self.model = tf.keras.models.load_model('./initial_model')  # Replace with your model initialization code
        self.client_weights = {}
    
    def fit(self, parameters):
        self.client_weights[parameters.client_id] = parameters.parameters
        if len(self.client_weights) == len(self.clients):
            # Aggregate client weights to update the global model
            global_weights = self.aggregate_weights(self.client_weights)
            self.model.set_weights(global_weights)
            self.client_weights = {}  # Clear client weights for the next round
        return fl.common.Responses.SUCCESS
    
    def evaluate(self, parameters):
        # You can implement evaluation logic here if needed
        return fl.common.Responses.SUCCESS

    def get_parameters(self):
        # Return the current global model parameters to the client
        return self.model.get_weights()

    def aggregate_weights(self, client_weights):
        # Aggregate client weights (e.g., simple averaging)
        num_clients = len(client_weights)
        if num_clients == 0:
            return self.model.get_weights()
        
        global_weights = np.zeros_like(self.model.get_weights())
        for weights in client_weights.values():
            global_weights += weights
        
        global_weights /= num_clients
        return global_weights
    
def fit_fn(metrics_list):
    try:
        loss = np.mean([metrics[0] for metrics in metrics_list])
        accuracy = np.mean([metrics[2]["accuracy"] for metrics in metrics_list])
        return {
            "loss": loss,
            "accuracy": accuracy,
        }
    except IndexError:
        # Handle the case where the tuple index is out of range
        return {
            "error": "Tuple index out of range",
        }


def evaluate_fn(metrics_list):
    try:
        loss = np.mean([metrics[0] for metrics in metrics_list])
        accuracy = np.mean([metrics[1]["accuracy"] for metrics in metrics_list])  # Use index 1 for num_examples
        return {
            "loss": loss,
            "accuracy": accuracy,
        }
    except IndexError:
        # Handle the case where the tuple index is out of range
        return {
            0,
        }




strategy = fl.server.strategy.FedAvg(
    fit_metrics_aggregation_fn=fit_fn,    
    evaluate_metrics_aggregation_fn=evaluate_fn,
)

if __name__ == "__main__":
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)
