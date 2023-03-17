# pylint: disable=too-many-lines,invalid-name
import warnings
import sys
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
import sklearn
#import mean absolute error from sklearn
from sklearn.metrics import mean_absolute_error as mae

np.random.seed(42)
random.seed(42)


class data_preprocess:

    def __init__(self, property = "H_atomization",Nmax=10000, binsize=3.0, avg_hydrogens=True) -> None:
        self.property = property
        self.Nmax = Nmax
        self.bin_size = binsize
        self.avg_hydrogens = avg_hydrogens


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

    def reduce_hydrogens(self,q,x):
        
        hydro_inds = q == 1
        x_heavy = x[np.argwhere(hydro_inds==False).flatten()]
        x_hydro = x[np.argwhere(hydro_inds==True).flatten()]
        x_avg_hydro =  np.vstack((x_heavy, np.mean(x_hydro,axis=0))).flatten()

        return x_avg_hydro

    def gen_rep(self):
        mbdf = MBDF.generate_mbdf(self.nuclear_charges,self.coords, n_jobs=-1,normalized = False, progress=False)
        X = []
        if self.avg_hydrogens:
            for q, x in zip(self.nuclear_charges, mbdf):
                X.append(self.reduce_hydrogens(q,x))
            X = np.array(X)
            max_size = max([len(subarray) for subarray in X])
            self.X = np.array([np.pad(subarray, (0, max_size - len(subarray)), 'constant') for subarray in X])
        else:
            self.X = MBDF.generate_DF(mbdf,self.nuclear_charges,  binsize=self.bin_size)

    def split_data(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.properties, test_size=0.2, random_state=42)
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test
        
    
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
        #pdb.set_trace()
        grid_search_concrete = GridSearchCV(ConcreteXGBRegressor(), param_grid, cv=n_folds, n_jobs=n_jobs)
        if N_max is not None:
            grid_search_concrete.fit(self.X_train[:N_max], self.y_train[:N_max])
        else:
            grid_search_concrete.fit(self.X_train, self.y_train)

        self.best_params_xgboost = grid_search_concrete.best_params_
        self.concrete_reg = grid_search_concrete.best_estimator_
        
    
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
    N_train = [2**i for i in range(5, 10)]

    hydros = [False, True]
    for h in hydros:

        X_train, X_test, y_train, y_test = data_preprocess(avg_hydrogens=h).run()
        

        fhe_boost = fhe_boost(X_train, y_train)
        for n in N_train:
            fhe_boost.cross_validation(N_max=n)
            y_pred_clear = fhe_boost.predict(X_test, execute_in_fhe=False)
            #y_pred_fhe = fhe_boost.predict(X_test, execute_in_fhe=True)
            MAE = mae(y_test, y_pred_clear)
            print(n, MAE)


    pdb.set_trace()
    #fhe_boost.quantize_model(N_max=100)
    print("Model trained and compiled.")

    # Let's instantiate the network
    network = OnDiskNetwork()
    fhemodel_dev = FHEModelDev(network.dev_dir.name, fhe_boost.concrete_reg)
    fhemodel_dev.save()
    print(os.listdir(network.dev_dir.name))
    network.dev_send_model_to_server()
    # Let's send the clientspecs and evaluation key to the client
    network.dev_send_clientspecs_and_modelspecs_to_client()
    #y_pred = fhe_boost.predict(X_test, execute_in_fhe=True)
    
    # Let's create the client and load the model
    fhemodel_client = FHEModelClient(network.client_dir.name, key_dir=network.client_dir.name)

    # The client first need to create the private and evaluation keys.
    fhemodel_client.generate_private_and_evaluation_keys()
    # Get the serialized evaluation keys
    serialized_evaluation_keys = fhemodel_client.get_serialized_evaluation_keys()
    print(f"Evaluation keys size: {sys.getsizeof(serialized_evaluation_keys) / 1024 / 1024:.2f} MB")
    # Let's send this evaluation key to the server (this has to be done only once)
    network.client_send_evaluation_key_to_server(serialized_evaluation_keys)
    print(os.listdir(network.server_dir.name))
    print(os.listdir(network.client_dir.name))
    print(os.listdir(network.dev_dir.name))
    pdb.set_trace()