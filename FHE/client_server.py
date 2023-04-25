import platform
import sys, os
import time
from shutil import copyfile
from tempfile import TemporaryDirectory
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import XGBClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np
import pdb

class OnDiskNetwork:
    """Simulate a network on disk."""

    def __init__(self):
        # Create 3 temporary folder for server, client and dev with tempfile
        self.server_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.client_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.dev_dir = TemporaryDirectory()  # pylint: disable=consider-using-with

    def client_send_evaluation_key_to_server(self, serialized_evaluation_keys):
        """Send the public key to the server."""
        with open(self.server_dir.name + "/serialized_evaluation_keys.ekl", "wb") as f:
            f.write(serialized_evaluation_keys)

    def client_send_input_to_server_for_prediction(self, encrypted_input):
        """Send the input to the server and execute on the server in FHE."""
        with open(self.server_dir.name + "/serialized_evaluation_keys.ekl", "rb") as f:
            serialized_evaluation_keys = f.read()
        time_begin = time.time()
        encrypted_prediction = FHEModelServer(self.server_dir.name).run(
            encrypted_input, serialized_evaluation_keys
        )
        time_end = time.time()
        with open(self.server_dir.name + "/encrypted_prediction.enc", "wb") as f:
            f.write(encrypted_prediction)
        return time_end - time_begin

    def dev_send_model_to_server(self):
        """Send the model to the server."""
        copyfile(self.dev_dir.name + "/server.zip", self.server_dir.name + "/server.zip")

    def server_send_encrypted_prediction_to_client(self):
        """Send the encrypted prediction to the client."""
        with open(self.server_dir.name + "/encrypted_prediction.enc", "rb") as f:
            encrypted_prediction = f.read()
        return encrypted_prediction

    def dev_send_clientspecs_and_modelspecs_to_client(self):
        """Send the clientspecs and evaluation key to the client."""
        copyfile(self.dev_dir.name + "/client.zip", self.client_dir.name + "/client.zip")
        copyfile(
            self.dev_dir.name + "/serialized_processing.json",
            self.client_dir.name + "/serialized_processing.json",
        )

    def cleanup(self):
        """Clean up the temporary folders."""
        self.server_dir.cleanup()
        self.client_dir.cleanup()
        self.dev_dir.cleanup()



# Let's first get some data and train a model.
X, y = load_breast_cancer(return_X_y=True)

# Split X into X_model_owner and X_client
X_model_owner, X_client = X[:-10], X[-10:]
y_model_owner, y_client = y[:-10], y[-10:]

# Some issues on macOS, if too many estimators
n_estimators = 10


# Train the model and compile it
model_dev = XGBClassifier(n_bits=2, n_estimators=n_estimators, max_depth=3)
model_dev.fit(X_model_owner, y_model_owner)
model_dev.compile(X_model_owner)



# Let's instantiate the network
network = OnDiskNetwork()

# Now that the model has been trained we want to save it to send it to a server
fhemodel_dev = FHEModelDev(network.dev_dir.name, model_dev)
#list all files with file size in network.dev_dir.name here
fhemodel_dev.save()
print(os.listdir(network.dev_dir.name))
network.dev_send_model_to_server()

# Let's send the clientspecs and evaluation key to the client
network.dev_send_clientspecs_and_modelspecs_to_client()


# Let's create the client and load the model
fhemodel_client = FHEModelClient(network.client_dir.name, key_dir=network.client_dir.name)

# The client first need to create the private and evaluation keys.
fhemodel_client.generate_private_and_evaluation_keys()
# Get the serialized evaluation keys
serialized_evaluation_keys = fhemodel_client.get_serialized_evaluation_keys()
     
# Evaluation keys can be quite large files but only have to be shared once with the server.
# Check the size of the evaluation keys (in MB)
print(f"Evaluation keys size: {sys.getsizeof(serialized_evaluation_keys) / 1024 / 1024:.2f} MB")

# Let's send this evaluation key to the server (this has to be done only once)
network.client_send_evaluation_key_to_server(serialized_evaluation_keys)

print(os.listdir(network.server_dir.name))
print(os.listdir(network.client_dir.name))
print(os.listdir(network.dev_dir.name))
#exit()
# Now we have everything for the client to interact with the server

# We create a loop to send the input to the server and receive the encrypted prediction
decrypted_predictions = []
execution_time = []
for i in range(X_client.shape[0]):
    clear_input = X_client[[i], :]
    encrypted_input = fhemodel_client.quantize_encrypt_serialize(clear_input)
    
    execution_time += [network.client_send_input_to_server_for_prediction(encrypted_input)]
    
    encrypted_prediction = network.server_send_encrypted_prediction_to_client()
    
    decrypted_prediction = fhemodel_client.deserialize_decrypt_dequantize(encrypted_prediction)[0]
    pdb.set_trace()
    decrypted_predictions.append(decrypted_prediction)

# Check MB size with sys of the encrypted data vs clear data
print(
    f"Encrypted data is "
    f"{sys.getsizeof(encrypted_input)/sys.getsizeof(clear_input):.2f}"
    " times larger than the clear data"
)

# Show execution time
print(
    f"Execution time are {[np.round(e, 2) for e in execution_time]}, ie an average of "
    f"{np.mean(execution_time):.2f} seconds"
)