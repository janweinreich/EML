# pylint: disable=too-many-lines,invalid-name
import warnings

# For warnings in xgboost.sklearn
warnings.simplefilter(action="ignore", category=FutureWarning)
#https://github.com/zama-ai/concrete-ml/blob/release/0.6.x/docs/advanced_examples/XGBRegressor.ipynb
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost.sklearn import XGBRegressor as SklearnXGBRegressor
import wget
from concrete.ml.sklearn import XGBRegressor as ConcreteXGBRegressor, Ridge
import random
import MBDF
import pdb
from tempfile import TemporaryDirectory
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from shutil import copyfile
np.random.seed(42)
random.seed(42)


class data_preprocess:

    def __init__(self, property = "H_atomization",Nmax=10000, binsize=3.0) -> None:
        self.property = property
        self.Nmax = Nmax
        self.bin_size = binsize


    def qm9_loader(self):
        if os.path.exists("qm9_data.npz"):
            qm9 = np.load("qm9_data.npz", allow_pickle=True)
        else:
            url = 'https://ndownloader.figshare.com/files/28893843'
            filename = wget.download(url)
            qm9 = np.load("qm9_data.npz", allow_pickle=True)

        qm9_inds = qm9["index"]
        coords = qm9['coordinates']
        nuclear_charges = qm9['charges']
        elements = qm9['elements']
        properties = np.array(qm9[self.property])
        N = len(qm9_inds)
        inds = np.arange(N)
        np.random.shuffle(inds)
        qm9_inds = qm9_inds[inds]
        coords = coords[inds]
        nuclear_charges = nuclear_charges[inds]
        elements = elements[inds]
        properties = properties[inds]

        self.qm9_inds = qm9_inds[:self.Nmax]
        self.coords = coords[:self.Nmax]
        self.nuclear_charges = nuclear_charges[:self.Nmax]
        self.elements = elements[:self.Nmax]
        self.properties = properties[:self.Nmax]

    def gen_rep(self):
        mbdf = MBDF.generate_mbdf(self.nuclear_charges,self.coords, n_jobs=-1,normalized = False, progress=False)
        pdb.set_trace()
        self.X = MBDF.generate_DF(mbdf,self.nuclear_charges,  binsize=self.bin_size)

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.properties, test_size=0.2, random_state=42)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    
    def run(self):
        self.qm9_loader()
        self.gen_rep()
        self.split_data()
        return self.X_train, self.X_test, self.y_train, self.y_test


class fhe_boost:
    def __init__(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train
        
    def cross_validation(self,N_max ):
        n_folds = 5
        n_jobs = -1
        param_grid = {
            "n_bits": [2, 3, 4, 5, 6, 7],
            "max_depth": [4],
            "n_estimators": [10, 20, 50, 100],
        }
        
        grid_search_concrete = GridSearchCV(ConcreteXGBRegressor(), param_grid, cv=n_folds, n_jobs=n_jobs)
        if N_max is not None:
            grid_search_concrete.fit(self.X_train[:N_max], self.y_train[:N_max])
        else:
            grid_search_concrete.fit(self.X_train, self.y_train)

        self.best_params_xgboost = grid_search_concrete.best_params_
        
    
    def quantize_model(self, N_max):
        self.concrete_reg = ConcreteXGBRegressor(**self.best_params_xgboost, n_jobs=-1)
        if N_max is not None:
            self.concrete_reg.fit(self.X_train[:N_max], self.y_train[:N_max])
            self.circuit = self.concrete_reg.compile(self.X_train[:N_max])
        else:
            self.concrete_reg.fit(self.X_train, self.y_train)
            self.circuit = self.concrete_reg.compile(self.X_train)
        print(f"Generating a key for an {self.circuit.graph.maximum_integer_bit_width()}-bits circuit")
        self.circuit.client.keygen(force=False)
    
    def predict(self, X_test, execute_in_fhe=True ):
        return self.concrete_reg.predict(X_test, execute_in_fhe=execute_in_fhe)


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


# main
if __name__ == "__main__":

    #Development Server
    X_train, X_test, y_train, y_test = data_preprocess().run()
    #pdb.set_trace()
    fhe_boost = fhe_boost(X_train, y_train)

    fhe_boost.cross_validation(N_max=100)
    fhe_boost.quantize_model(N_max=100)
    print("Model trained and compiled.")

    # Let's instantiate the network
    network = OnDiskNetwork()
    fhemodel_dev = FHEModelDev(network.dev_dir.name, fhe_boost.concrete_reg)
    fhemodel_dev.save()
    print(os.listdir(network.dev_dir.name))
    pdb.set_trace()
    #y_pred = fhe_boost.predict(X_test, execute_in_fhe=True)
    