import os


import pandas as pd
import numpy as np

import flwr as fl
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Load the SavedModel
model = tf.keras.models.load_model('./initial_model')

# Convert the SavedModel to a Keras model
# model = tf.keras.models.Sequential()
# model = loaded_model.signatures["serving_default"]

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the CSV file
data = pd.read_csv('data/client_data_three.csv')

# Extract the features (x) and labels (y) from the data
X = data.drop(['url','type','Category','domain'],axis=1)#,'type_code'
y = data['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)



# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(X_train, y_train, epochs=1, batch_size=32)
     
        # Return updated model parameters and results
        parameters_prime = model.get_weights()
        num_examples_train = len(X_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            
        }
        return parameters_prime, num_examples_train, results
        # return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())
