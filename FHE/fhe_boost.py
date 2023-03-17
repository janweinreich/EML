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
        self.X    = MBDF.generate_DF(mbdf,self.nuclear_charges,  binsize=self.bin_size)

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



# main
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_preprocess().run()
    fhe_boost = fhe_boost(X_train, y_train)

    fhe_boost.cross_validation(N_max=100)
    fhe_boost.quantize_model(N_max=100)
    pdb.set_trace()
    y_pred = fhe_boost.predict(X_test, execute_in_fhe=True)
    