import warnings
import sys
# For warnings in xgboost.sklearn
warnings.simplefilter(action="ignore", category=FutureWarning)
#https://github.com/zama-ai/concrete-ml/blob/release/0.6.x/docs/advanced_examples/XGBRegressor.ipynb
import time
import os
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV, train_test_split
import wget
from concrete.ml.sklearn import XGBRegressor as ConcreteXGBRegressor, Ridge, Lasso
import random
import MBDF

import pdb
from tempfile import TemporaryDirectory
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from shutil import copyfile
from tqdm import tqdm
from qstack import compound, spahm
from sklearn.metrics import mean_absolute_error as mae
import socket
import threading
from shutil import copyfile
from tempfile import TemporaryDirectory
import pickle
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)


def dump_to_file(obj, file_path):
    """Dump a Python object to a file using pickle."""
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_from_file(file_path):
    """Load a Python object from a file using pickle."""
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def mol_to_xyz(els, coords, filename="curr.xyz"):
    #if not exist tmp dir create it
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    path_to_temp_xyz = os.path.join("tmp", filename)
    with open(path_to_temp_xyz, 'w') as f:
        f.write(str(len(els)) + "\n")
        f.write("\n")
        for el, coord in zip(els, coords):
            f.write(str(el) + " {} {} {}\n".format(*coord))
    f.close()
    return path_to_temp_xyz



def max_element_counts(elements):
    # Initialize an empty dictionary to store the maximum counts
    max_counts = {}

    # Iterate over each sub-array in the main array
    for sub_array in elements:
        # Count the occurrences of each element in the sub-array
        unique, counts = np.unique(sub_array, return_counts=True)
        count_dict = dict(zip(unique, counts))

        # Compare the counts with the current maximum counts
        for element, count in count_dict.items():
            if element not in max_counts or count > max_counts[element]:
                max_counts[element] = count

    return max_counts


class Data_preprocess:

    def __init__(self, property = "H_atomization",N_max="all", binsize=3.0,rep_type="spahm", avg_hydrogens=False) -> None:
        self.property = property
        self.N_max = N_max
        self.rep_type = rep_type
        self.bin_size = binsize
        self.avg_hydrogens = avg_hydrogens


    def qm9_loader(self):
        if os.path.exists("qm9_data.npz"):
            qm9 = np.load("qm9_data.npz", allow_pickle=True)
        else:
            url = 'https://ndownloader.figshare.com/files/28893843'
            qm9 = np.load("qm9_data.npz", allow_pickle=True)

        qm9_inds = qm9["index"]
        coords = qm9['coordinates']
        nuclear_charges = qm9['charges']
        elements = qm9['elements']
        properties = np.array(qm9[self.property])
        N = len(qm9_inds)

        if self.N_max == "all":
            self.N_max = N

        inds = np.arange(N)
        np.random.shuffle(inds)
        qm9_inds = qm9_inds[inds]
        coords = coords[inds]
        nuclear_charges = nuclear_charges[inds]
        elements = elements[inds]
        properties = properties[inds]
        self.qm9_inds = qm9_inds[:self.N_max]
        self.coords = coords[:self.N_max]
        self.nuclear_charges = nuclear_charges[:self.N_max]
        self.elements = elements[:self.N_max]
        self.properties = properties[:self.N_max]

        


    def reduce_hydrogens(self,q,x):
        
        hydro_inds = q == 1
        x_heavy = x[np.argwhere(hydro_inds==False).flatten()]
        x_hydro = x[np.argwhere(hydro_inds==True).flatten()]
        x_avg_hydro =  np.vstack((x_heavy, np.mean(x_hydro,axis=0))).flatten()

        return x_avg_hydro

    def gen_rep(self):
        
        if "mbdf" in self.rep_type:

            mbdf = MBDF.generate_mbdf(self.nuclear_charges,self.coords, n_jobs=-1) #,normalized = True) #, progress=False)
            #pdb.set_trace()

            
            if self.rep_type=="local_mbdf":
                if self.avg_hydrogens:
                    X = []
                    for q, x in zip(self.nuclear_charges, mbdf):
                        X.append(self.reduce_hydrogens(q,x))
                    X = np.array(X)

                    self.X = self.pad_max_size(X)
                else:
                    self.X =  np.array([x.flatten() for x in mbdf])
            #TODO: sum over mbdf over 1st axis instead of flattening!!!
            elif self.rep_type == "local_mbdf_order":
                row_norms = np.linalg.norm(mbdf, axis=2)
                sorted_indices = np.argsort(-row_norms)
                sorted_mbdf = np.take_along_axis(mbdf, sorted_indices[:, :, np.newaxis], axis=1)
                
                self.X = np.array([x.flatten() for x in sorted_mbdf])

            elif self.rep_type == "local_2_global_mbdf":
                row_norms = np.linalg.norm(mbdf, axis=2)
                sorted_indices = np.argsort(-row_norms)
                sorted_mbdf = np.take_along_axis(mbdf, sorted_indices[:, :, np.newaxis], axis=1)
                #pdb.set_trace()
                self.X = np.concatenate([np.sum(sorted_mbdf, axis=2),np.mean(sorted_mbdf, axis=2), np.var(sorted_mbdf, axis=2)], axis=1)
                #np.sum(sorted_mbdf, axis=2)


            elif self.rep_type == "global_mbdf":
                self.X = MBDF.generate_df(mbdf,self.nuclear_charges,  binsize=self.bin_size)

            # todo bagging mbdf? slighlty better than BoB but not as good as global mbdf, after creating bags pad with zeros to max size of bags and then order within bags by norm
            # global mbdf is best and is contant size for all molecules (no padding needed)
            # Molecular feature based representation, histogram of elements, partial charges, etc
            # try lasso regression 


        
        elif self.rep_type == "spahm":
            print("Generating spahm representations...")
            
            
            X = []
            for els, coord in zip(self.elements, self.coords): #, total=len(self.coords)):
                mol_name = mol_to_xyz(els, coord)
                
                mol = compound.xyz_to_mol(mol_name, 'def2svp', charge=0, spin=0)
                os.remove(mol_name)
                X.append(spahm.compute_spahm.get_spahm_representation(mol, "lb")[0])
            
            self.X = X
            self.X = self.pad_max_size(self.X)
            self.X = np.array(self.X)

        elif self.rep_type == "BoB":
            self.X = MBDF.generate_bob(self.elements,self.coords, asize=max_element_counts(self.elements))
            
        elif self.rep_type == "MBDF_bagged":
            import MBDF_BAGGED
            self.X = MBDF_BAGGED.generate_mbdf_bagged(self.nuclear_charges,self.coords,asize = max_element_counts(self.elements))


        else:
            raise ValueError("rep_type must be either local_mbdf,BoB, global_mbdf or spahm!")


    def pad_max_size(self, X):
        max_size = max([len(subarray) for subarray in X])
        X= np.array([np.pad(subarray, (0, max_size - len(subarray)), 'constant') for subarray in X])
        return X

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




class Fhe_boost:
    def __init__(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train
        
    def cross_validation(self,N_max ):
        n_folds = 5
        n_jobs = -1
        param_grid = {
            "n_bits": [6, 7, 10],
            "max_depth": [5,6, 8, 10, 12, 13],
            "n_estimators": [10,15, 20, 25, 50],
        }

        grid_search_concrete = GridSearchCV(ConcreteXGBRegressor(), param_grid, cv=n_folds, n_jobs=n_jobs)
        if N_max is not None:
            grid_search_concrete.fit(self.X_train[:N_max], self.y_train[:N_max])
        else:
            grid_search_concrete.fit(self.X_train, self.y_train)

        self.reg_best_params = grid_search_concrete.best_params_
        self.concrete_reg = grid_search_concrete.best_estimator_
        self.concrete_reg.compile(self.X_train)

    def train(self, N_max,params):
        self.concrete_reg = ConcreteXGBRegressor(**params, n_jobs=-1)
        if N_max is not None:
            self.concrete_reg.fit(self.X_train[:N_max], self.y_train[:N_max])
        else:
            self.concrete_reg.fit(self.X_train, self.y_train)
        
    
    def quantize_model(self, N_max):
        #self.concrete_reg = ConcreteXGBRegressor(**self.best_params_xgboost, n_jobs=-1)
        if N_max is not None:
            #self.concrete_reg.fit(self.X_train[:N_max], self.y_train[:N_max])
            self.circuit = self.concrete_reg.compile(self.X_train[:N_max])
        else:
            #self.concrete_reg.fit(self.X_train, self.y_train)
            self.circuit = self.concrete_reg.compile(self.X_train)
        print(f"Generating a key for an {self.circuit.graph.maximum_integer_bit_width()}-bits circuit")
        self.circuit.client.keygen(force=False)
    
    def predict(self, X_test, fhe=True):
        if fhe:
            return self.concrete_reg.predict(X_test, fhe="execute")
                                         #execute_in_fhe=execute_in_fhe)
        else: 
            return self.concrete_reg.predict(X_test, fhe="simulate")
    

class Fhe_ridge:
    def __init__(self, X_train, y_train, clear_model="Ridge") -> None:
        self.X_train = X_train
        self.y_train = y_train
        if clear_model == "Ridge":
            self.clear_model = Ridge
        elif clear_model == "Lasso":
            self.clear_model = Lasso

    def cross_validation(self,N_max ):

        n_folds = 5
        n_jobs = -1
        #param_grid = {
        #    "n_bits": [12, 13, 15, 16, 18],
        #    "alpha": [1e-8,1e-7, 1e-6,1e-5,1e-4, 1e-3], with these parameters we get a compilation error
        #}
        param_grid = { "n_bits": [11,12], "alpha": [1e-8,1e-7, 1e-6,1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}

        grid_search_concrete = GridSearchCV(self.clear_model(), param_grid, cv=n_folds, n_jobs=n_jobs)
        if N_max is not None:
            grid_search_concrete.fit(self.X_train[:N_max], self.y_train[:N_max])
        else:
            grid_search_concrete.fit(self.X_train, self.y_train)

        self.reg_best_params = grid_search_concrete.best_params_
        print(self.reg_best_params)
        self.concrete_reg = grid_search_concrete.best_estimator_
        #pdb.set_trace()
        self.concrete_reg.compile(self.X_train)

    def train(self,N_max, params):
        self.concrete_reg = self.clear_model(**params)
        if N_max is not None:
            self.concrete_reg.fit(self.X_train[:N_max], self.y_train[:N_max])
        else:
            self.concrete_reg.fit(self.X_train, self.y_train)
    
    def quantize_model(self, N_max):
        if N_max is not None:
            self.circuit = self.concrete_reg.compile(self.X_train[:N_max])
        else:
            self.circuit = self.concrete_reg.compile(self.X_train)
        print(f"Generating a key for an {self.circuit.graph.maximum_integer_bit_width()}-bits circuit")
        self.circuit.client.keygen(force=False)
    
    def predict(self, X_test, fhe=True):
        if fhe:
            return self.concrete_reg.predict(X_test, fhe="execute")
                                         #execute_in_fhe=execute_in_fhe)
        else:
            return self.concrete_reg.predict(X_test, fhe="simulate")

    

class Test_fhe():
    def __init__(self, regressor, outname) -> None:
        self.regressor = regressor
        self.binsizes  = np.linspace(0.1, 3.0, 10)
        self.outname = outname



    
    def rep_len(self):
        self.n_train_max = 128
        #binsize make intervals of 0.1 between 0.1 and 3.0 wiht linspace
        
        self.test_rep_len_results = {}

        rep_shapes, fhe_times,clear_times, fhe_mae, clear_mae = [], [], [], [], []
        for b in self.binsizes:
            X_train, X_test, y_train, y_test = Data_preprocess(binsize=b, avg_hydrogens=False, N_max=3000).run()
            X_test, y_test = X_test[:10], y_test[:10]
            repshape = X_train.shape[1]
            fhe_instance = self.regressor(X_train, y_train)
            if self.regressor == Fhe_boost:
                fhe_instance.train(N_max=self.n_train_max, params = {"n_bits": 3, "max_depth": 4, "n_estimators": 10})
            elif self.regressor == Fhe_ridge:
                fhe_instance.train(N_max=self.n_train_max, params = {"n_bits": 3, "alpha": 1e-3})
            else:
                raise ValueError("Regressor not supported")
            fhe_instance.quantize_model(N_max=self.n_train_max)

            time_begin = time.time()
            y_pred_fhe = []
            for i in tqdm(range(len(X_test))):
                y_pred_fhe.append(fhe_instance.predict(X_test[i].reshape(1,-1), fhe=True)[0][0])
            runtime_fhe = (time.time() - time_begin)/len(X_test)
            y_pred_fhe = np.array(y_pred_fhe)

            time_begin = time.time()
            y_pred_clear = fhe_instance.predict(X_test)
            runtime_clear = (time.time() - time_begin)/len(X_test)
            print(f"Runtime per sample FHE : {(runtime_fhe):.2f} sec")
            print(f"Runtime per sample clear: {(runtime_clear):.2f} sec")

            MAE_fhe = mae(y_test, y_pred_fhe)
            MAE_clear = mae(y_test, y_pred_clear)
            print(b, repshape,runtime_fhe,runtime_clear, MAE_fhe, MAE_clear)

            rep_shapes.append(repshape)
            fhe_times.append(runtime_fhe)
            clear_times.append(runtime_clear)
            fhe_mae.append(MAE_fhe)
            clear_mae.append(MAE_clear)

        
        rep_shapes, fhe_times,clear_times, fhe_mae, clear_mae = np.array(rep_shapes), np.array(fhe_times), np.array(clear_times), np.array(fhe_mae), np.array(clear_mae)
        self.test_rep_len_results["rep_shapes"] = rep_shapes
        self.test_rep_len_results["fhe_times"] = fhe_times
        self.test_rep_len_results["clear_times"] = clear_times
        self.test_rep_len_results["fhe_mae"] = fhe_mae
        self.test_rep_len_results["clear_mae"] = clear_mae
        self.test_rep_len_results["binsizes"] = self.binsizes
        self.test_rep_len_results["n_train_max"] = self.n_train_max
    

    def local_hydro_averaging(self):
        #Test model with hydrogen averaging and without
        self.test_hydro_averaging_results = {}
        hydros = [False, True]
        for h in hydros:

            X_train, X_test, y_train, y_test = Data_preprocess(rep_type="local_mbdf", avg_hydrogens=h, N_max=3000).run()
            print("Rep dim", X_train[0].shape)
            learning_curve = []
            fhe_instance = self.regressor(X_train, y_train)
            for n in self.N_train:
                fhe_instance.cross_validation(N_max=n)
                y_pred_clear = fhe_instance.predict(X_test, fhe=False)
                MAE = mae(y_test, y_pred_clear)
                print(n, MAE)
                learning_curve.append(MAE)
            
            curr_results = {}
            curr_results["learning_curve"] = learning_curve
            curr_results["rep_shape"] = X_train.shape[1]


            if h:
                self.test_hydro_averaging_results["hydro"] = curr_results
            else:
                self.test_hydro_averaging_results["no_hydro"] = curr_results

        self.test_hydro_averaging_results["N_train"] = self.N_train

    def local_order(self, clear_model="Ridge"):
        #Test model with mbdf and without
        self.test_local_order_results = {}
        X_train, X_test, y_train, y_test = Data_preprocess(rep_type="local_mbdf_order", avg_hydrogens=False, N_max=1500).run()
        print("Rep dim", X_train[0].shape)
        n_max = X_train.shape[0]
        max_power = int(np.log2(n_max)) 
        curve = np.logspace(5, max_power, max_power-1, base=2, dtype=int)
        self.N_train =  np.append(curve, n_max)

        learning_curve = []
        fhe_instance = self.regressor(X_train, y_train, clear_model)
        for n in self.N_train:
            fhe_instance.cross_validation(N_max=n)
            y_pred_clear = fhe_instance.predict(X_test, fhe=False)
            MAE = mae(y_test, y_pred_clear)
            print(n, MAE)
            learning_curve.append(MAE)
        
        self.test_local_order_results["learning_curve"] = learning_curve
        self.test_local_order_results["rep_shape"] = X_train.shape[1]
        self.test_local_order_results["N_train"] = self.N_train


    def local_global(self, clear_model="Ridge"):
        self.test_local_order_results = {}
        X_train, X_test, y_train, y_test = Data_preprocess(rep_type="local_2_global_mbdf", avg_hydrogens=False, N_max=1500).run()
        print("Rep dim", X_train[0].shape)
        n_max = X_train.shape[0]
        max_power = int(np.log2(n_max)) 
        curve = np.logspace(5, max_power, max_power-1, base=2, dtype=int)
        self.N_train =  np.append(curve, n_max)
        learning_curve = []
        fhe_instance = self.regressor(X_train, y_train, clear_model)
        for n in self.N_train:
            fhe_instance.cross_validation(N_max=n)
            y_pred_clear = fhe_instance.predict(X_test, fhe=False)
            MAE = mae(y_test, y_pred_clear)
            print(n, MAE)
            learning_curve.append(MAE)
        
        self.test_local_order_results["learning_curve"] = learning_curve
        self.test_local_order_results["rep_shape"] = X_train.shape[1]
        self.test_local_order_results["N_train"] = self.N_train


        return fhe_instance, self.N_train, learning_curve, len(X_train[0])



    def BoB_global(self, fhe=False, clear_model="Ridge"):
        self.test_BoB_global = {}
        X_train, X_test, y_train, y_test = Data_preprocess(rep_type="BoB", avg_hydrogens=False, N_max=1500).run()
        print("Rep dim", X_train[0].shape)
        n_max = X_train.shape[0]
        max_power = int(np.log2(n_max)) 
        curve = np.logspace(5, max_power, max_power-1, base=2, dtype=int)
        self.N_train =  np.append(curve, n_max)
        learning_curve = []
        fhe_instance = self.regressor(X_train, y_train, clear_model)
        for n in self.N_train:
            fhe_instance.cross_validation(N_max=n)
            y_pred_clear = fhe_instance.predict(X_test, fhe=False)
            MAE = mae(y_test, y_pred_clear)
            if fhe:
                y_pred_fhe = fhe_instance.predict(X_test, fhe=True)
                MAE_fhe = mae(y_test, y_pred_fhe)
            
                print(n,fhe_instance.reg_best_params, MAE, MAE_fhe)
            else:
                print(n,fhe_instance.reg_best_params, MAE)
            learning_curve.append(MAE)
        
        return fhe_instance, self.N_train, learning_curve, len(X_train[0])

        

    def MBDF_bagged_rep(self, fhe=False, clear_model="Ridge"):
        X_train, X_test, y_train, y_test = Data_preprocess(rep_type="MBDF_bagged", avg_hydrogens=False, N_max=1500).run()
        print("Rep dim", len(X_train[0]))
        #pdb.set_trace()
        n_max = X_train.shape[0]
        max_power = int(np.log2(n_max)) 
        curve = np.logspace(5, max_power, max_power-1, base=2, dtype=int)
        self.N_train =  np.append(curve, n_max)
        learning_curve = []
        fhe_instance = self.regressor(X_train, y_train, clear_model)
        for n in self.N_train:
            fhe_instance.cross_validation(N_max=n)
            y_pred_clear = fhe_instance.predict(X_test, fhe=False)
            MAE = mae(y_test, y_pred_clear)
            if fhe:
                y_pred_fhe = fhe_instance.predict(X_test, fhe=True)
                MAE_fhe = mae(y_test, y_pred_fhe)
            
                print(n,fhe_instance.reg_best_params, MAE, MAE_fhe)
            else:
                print(n,fhe_instance.reg_best_params, MAE)
            learning_curve.append(MAE)
        
        return fhe_instance, self.N_train, learning_curve, len(X_train[0])




    def mbdf_global(self, fhe=False, clear_model="Ridge"):
        binsize = 0.05
        #Test model with mbdf and without
        self.test_mbdf_results = {}
        X_train, X_test, y_train, y_test = Data_preprocess(binsize=binsize, rep_type="global_mbdf", avg_hydrogens=False, N_max=1500).run()
        print("Rep dim", X_train[0].shape)
        n_max = X_train.shape[0]
        max_power = int(np.log2(n_max)) 
        curve = np.logspace(5, max_power, max_power-1, base=2, dtype=int)
        self.N_train =  np.append(curve, n_max)
        learning_curve = []
        fhe_instance = self.regressor(X_train, y_train, clear_model)
        #pdb.set_trace()
        for n in self.N_train:
            fhe_instance.cross_validation(N_max=n)
            y_pred_clear = fhe_instance.predict(X_test, fhe=False)
            MAE = mae(y_test, y_pred_clear)
            if fhe:
                y_pred_fhe = fhe_instance.predict(X_test, fhe=True)
                MAE_fhe = mae(y_test, y_pred_fhe)
            
                print(n,fhe_instance.reg_best_params, MAE, MAE_fhe)
            else:
                print(n,fhe_instance.reg_best_params, MAE)
            learning_curve.append(MAE)
        
        self.test_mbdf_results["learning_curve"] = learning_curve
        self.test_mbdf_results["rep_shape"] = X_train.shape[1]
        self.test_mbdf_results["binsize"] = binsize
        self.test_mbdf_results["N_train"] = self.N_train

        return fhe_instance, self.N_train, learning_curve, len(X_train[0])


    def spahm_global(self, clear_model="Ridge"):
        self.test_spahm_global_results = {}
        X_train, X_test, y_train, y_test = Data_preprocess(rep_type="spahm", avg_hydrogens=False,N_max=1500).run()
        print("Rep dim", X_train[0].shape)
        n_max = X_train.shape[0]
        max_power = int(np.log2(n_max)) 
        curve = np.logspace(5, max_power, max_power-1, base=2, dtype=int)
        self.N_train =  np.append(curve, n_max)
        
        
        learning_curve = []
        fhe_instance = self.regressor(X_train, y_train, clear_model)
        for n in self.N_train:
            #pdb.set_trace()
            fhe_instance.cross_validation(N_max=n)
            y_pred_clear = fhe_instance.predict(X_test, fhe=False)
            MAE = mae(y_test, y_pred_clear)
            print(n,fhe_instance.reg_best_params, MAE)
            learning_curve.append(MAE)

        self.test_spahm_global_results["learning_curve"] = learning_curve
        self.test_spahm_global_results["rep_shape"] = X_train.shape[1]
        self.test_spahm_global_results["N_train"] = self.N_train

        return fhe_instance, self.N_train, learning_curve, len(X_train[0])


    def save_results(self):
        """
        Method to save this instance of the test class to pkl file
        """
        with open(f"{self.outname}_results.pkl", "wb") as f:
            pickle.dump(self, f)



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




# main
if __name__ == "__main__":
    do_ridge, do_boost = True, False

    if do_ridge:
        test_class_ridge = Test_fhe(Fhe_ridge, "ridge")



        print("RIDGE")

        print("MBDF bagged")
        _, Ntrain, error_bagged, d_bagged =  test_class_ridge.MBDF_bagged_rep(fhe=False, clear_model="Ridge")
        
        print("BoB_global")
        _, Ntrain, global_error_bob, d_bob =  test_class_ridge.BoB_global(fhe=False, clear_model="Ridge")


        print("MBDF global")
        _, Ntrain, global_error_ridge, d_global =  test_class_ridge.mbdf_global(fhe=False, clear_model="Ridge")

        print("Local order")
        _,Ntrain, local_error_ridge, _ =  test_class_ridge.local_global(clear_model="Ridge")

        
        print("SPAHM global")
        _, Ntrain,spahm_error_ridge, d_spahm =  test_class_ridge.spahm_global(clear_model="Ridge")

        print("LASSO")
        fhe_instance, Ntrain, global_error_bob_lasso, _ =  test_class_ridge.BoB_global(fhe=False, clear_model="Lasso")


        print("MBDF global")
        fhe_instance, Ntrain, global_error_lasso, _ =  test_class_ridge.mbdf_global(fhe=False, clear_model="Lasso")

        print("Local order")
        _,Ntrain, local_error_lasso, _ =  test_class_ridge.local_global(clear_model="Lasso")

        print("SPAHM global")
        _, Ntrain,spahm_error_lasso, _ =  test_class_ridge.spahm_global(clear_model="Lasso")





        #plot the learning curves
        fig, ax = plt.subplots(figsize=(6, 8))
        #first plot the results with straight lines
        ax.plot(Ntrain, error_bagged, label="MBDF bagged", color="orange", linestyle="-", linewidth=3)
        ax.plot(Ntrain, global_error_bob, label="BoB global", color="black", linestyle="-", linewidth=3)
        ax.plot(Ntrain, global_error_ridge, label="MBDF global", color="blue", linestyle="-", linewidth=3)
        ax.plot(Ntrain, local_error_ridge, label="Local order", color="red", linestyle="-", linewidth=3)
        ax.plot(Ntrain, spahm_error_ridge, label="SPAHM global", color="green", linestyle="-", linewidth=3)
        #plot now the lasso results with dashed lines
        ax.plot(Ntrain, global_error_bob_lasso, color="black", linestyle="--", linewidth=3, alpha=0.3)
        ax.plot(Ntrain, global_error_lasso,  color="blue", linestyle="-", linewidth=3, alpha=0.3)
        ax.plot(Ntrain, local_error_lasso, color="red", linestyle="-", linewidth=3, alpha=0.3)
        ax.plot(Ntrain, spahm_error_lasso, color="green", linestyle="-", linewidth=3, alpha=0.3)


        ax.set_xlabel(r"$N$", fontsize=25)
        ax.set_xticks(Ntrain)
        ax.set_ylabel(r"MAE (ha)", fontsize=25)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(fontsize=12)

        fig.savefig("./tmp/learning_curves.pdf", bbox_inches="tight")


        fig2, ax2 = plt.subplots(figsize=(6, 8))
        #plot single points
        ax2.scatter(d_bagged, error_bagged[-1], label="MBDF bagged", color="orange", s=100)
        ax2.scatter(d_bob, global_error_bob[-1], label="BoB global", color="black", s=100)
        ax2.scatter(d_global, global_error_ridge[-1], label="MBDF global", color="blue", s=100)
        ax2.scatter(d_spahm, spahm_error_ridge[-1], label="SPAHM global", color="green", s=100)

        ax2.set_xlabel(r"$d$", fontsize=25)
        ax2.set_ylabel(r"MAE (ha)", fontsize=25)
        ax2.legend(fontsize=12)
        fig2.savefig("./tmp/pareto.pdf", bbox_inches="tight")

        exit()


    if do_boost:
        print("Boost")
        test_class_boost = Test_fhe(Fhe_boost, "boost")
        print("MBDF global")
        test_class_boost.mbdf_global()
        print("Local hydro averaging")
        test_class_boost.local_hydro_averaging()
        print("SPAHM global")
        test_class_boost.spahm_global()
        test_class_boost.save_results()



    #fhe_boost.quantize_model(N_max=100)
    print("Model trained and compiled.")

    local_net, online_net = False, True




    model_dev  = fhe_instance.concrete_reg
    
    if local_net:
        network = OnDiskNetwork()
        fhemodel_dev = FHEModelDev(network.dev_dir.name, model_dev)
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
        


        # Now we have everything for the client to interact with the server
        X_client = test_class_ridge.X_test[:10]
        # We create a loop to send the input to the server and receive the encrypted prediction
        decrypted_predictions = []
        execution_time = []
        for i in tqdm(range(X_client.shape[0])):
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
        print(f"The average execution time is {np.mean(execution_time):.2f} seconds per sample.")
        
        
        clear_prediction = model_dev.predict(X_client)
        decrypted_predictions = np.array(decrypted_predictions)
        deviation = mae(clear_prediction, decrypted_predictions)
        print(f"Deviation between FHE prediction and clear model is: {deviation}")

    else:
        network = OnlineNetwork()
        fhemodel_dev = FHEModelDev(network.dev_dir.name, model_dev)
        fhemodel_dev.save()
        print(os.listdir(network.server_dir.name))

        pdb.set_trace()
        server_thread = threading.Thread(target=network.start_server)
        server_thread.start()
        network.dev_send_model_to_server()




        
        server_thread.join()