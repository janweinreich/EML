import platform
import time
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy
from sklearn.datasets import load_breast_cancer

from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import XGBClassifier

from backend import OnDiskNetwork

# Let's create the client and load the model


# When starting the client, we need to know the server's IP address and port
# of course we 
fhemodel_client = FHEModelClient(network.client_dir.name, key_dir=network.client_dir.name)

# The client first need to create the private and evaluation keys.
fhemodel_client.generate_private_and_evaluation_keys()

# Get the serialized evaluation keys
serialized_evaluation_keys = fhemodel_client.get_serialized_evaluation_keys()


# Evaluation keys can be quite large files but only have to be shared once with the server.

# Check the size of the evaluation keys (in MB)
print(f"Evaluation keys size: {len(serialized_evaluation_keys) / (10**6):.2f} MB")

# Let's send this evaluation key to the server (this has to be done only once)
network.client_send_evaluation_key_to_server(serialized_evaluation_keys)


# Let's check the results and compare them against the clear model
clear_prediction_classes = model_dev.predict_proba(X_client).argmax(axis=1)
decrypted_predictions_classes = numpy.array(decrypted_predictions).argmax(axis=1)
accuracy = (clear_prediction_classes == decrypted_predictions_classes).mean()
print(f"Accuracy between FHE prediction and clear model is: {accuracy*100:.0f}%")