import platform
import time
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy
from sklearn.datasets import load_breast_cancer

from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import XGBClassifier

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

    def cleanup(self):
        """Clean up the temporary folders."""
        self.server_dir.cleanup()
        self.client_dir.cleanup()
        self.dev_dir.cleanup()
