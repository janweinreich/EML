import platform
import time
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy
from sklearn.datasets import load_breast_cancer

from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import XGBClassifier

from backend import OnDiskNetwork

# Let's first get some data and train a model.
X, y = load_breast_cancer(return_X_y=True)

# Split X into X_model_owner and X_client
X_model_owner, X_client = X[:-10], X[-10:]
y_model_owner, y_client = y[:-10], y[-10:]

# Some issues on macOS, if too many estimators
n_estimators = 10
if platform.system() == "Darwin":
    n_estimators = 9

# Train the model and compile it
model_dev = XGBClassifier(n_bits=2, n_estimators=n_estimators, max_depth=3)
model_dev.fit(X_model_owner, y_model_owner)
model_dev.compile(X_model_owner)

print("Model trained and compiled.")


# Let's instantiate the network
network = OnDiskNetwork()

# Now that the model has been trained we want to save it to send it to a server
fhemodel_dev = FHEModelDev(network.dev_dir.name, model_dev)
fhemodel_dev.save()


# Let's send the model to the server
network.dev_send_model_to_server()


# Let's send the clientspecs and evaluation key to the client
# First time we have to interact with the client
network.dev_send_clientspecs_and_modelspecs_to_client()


############################################
# Now wait for the client to send the input data and evaluation keys
############################################


# Now we have everything for the client to interact with the server
decrypted_predictions = []
execution_time = []
for i in range(X_client.shape[0]):
    clear_input = X_client[[i], :]
    encrypted_input = fhemodel_client.quantize_encrypt_serialize(clear_input)
    execution_time += [network.client_send_input_to_server_for_prediction(encrypted_input)]
    encrypted_prediction = network.server_send_encrypted_prediction_to_client()
    decrypted_prediction = fhemodel_client.deserialize_decrypt_dequantize(encrypted_prediction)[0]
    decrypted_predictions.append(decrypted_prediction)

# Check MB size with sys of the encrypted data vs clear data
print(
    f"Encrypted data is "
    f"{len(encrypted_input)/clear_input.nbytes:.2f}"
    " times larger than the clear data"
)

# Show execution time
print(f"The average execution time is {numpy.mean(execution_time):.2f} seconds per sample.")