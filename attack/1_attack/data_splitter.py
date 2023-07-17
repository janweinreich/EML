# general and data handling
import numpy as np
import pandas as pd
import os
import random
import rdkit as rd
import torch
import pdb
from rdkit.Chem import AllChem
import copy
import random
import math

random.seed(1336)

torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True
# number of bits for morgan fingerprints
morgan_bits,morgan_radius = 4096, 2

def SMILES2CANONICAL(data):
    # Convert SMILES to CANONICAL SMILES
    # In the process of canonicalizing SMILES, any bad SMILES definition
    #     is caught and removed from the dataset
    for i in range(len(data)):
        smis = list(data[i].smiles)

        cans = []
        for smi in smis:
            mol = rd.Chem.MolFromSmiles(smi)
            # see whether can be parsed to mol
            if mol:
                can = rd.Chem.MolToSmiles(mol, True)
                cans.append(can)
            else:
                cans.append(np.nan)

        data[i]['SMILES'] = cans
        # drop data point that has invalid molecule
        data[i] = data[i][data[i]['SMILES'].notna()]    
    
    return data

def split(dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
        Splits internal compounds randomly into train/validation/test.
        """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if not seed is None:
        np.random.seed(seed)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    shuffled = np.random.permutation(range(num_datapoints))
    return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff],
            shuffled[valid_cutoff:])


def apply_splitting(data):
    temp_train_data = []
    temp_test_data  = []
    temp_valid_data = []
    for i in range(len(data)):
        splitter_i = split(data[i])
        
        for j in range(len(splitter_i)):
                if j==0:
                    temp_train_data.append(data[i].iloc[splitter_i[j]])
                if j==1:
                    temp_test_data.append(data[i].iloc[splitter_i[j]])
                if j==2:
                    temp_valid_data.append(data[i].iloc[splitter_i[j]])

    train_data = temp_train_data[0]
    test_data  = temp_test_data[0]
    valid_data = temp_valid_data[0]

    for i in range(1, len(data)):
        train_data = train_data.merge(temp_train_data[i], how='outer', on='smiles')
        test_data  = test_data.merge(temp_test_data[i], how='outer', on='smiles')
        valid_data  = valid_data.merge(temp_valid_data[i], how='outer', on='smiles')
    data = [train_data, test_data, valid_data]
    return data

def save_data(data, data_path):
    torch.save(data[0], data_path+"train.pth")
    torch.save(data[1], data_path+"test.pth")
    torch.save(data[2], data_path+"valid.pth")


def check_float_not_nan(num):
    return isinstance(num, float) and not math.isnan(num)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tox21_file  = 'tox21.csv'
tox21_tasks = ['SR-ARE']


tox21_data = pd.read_csv(tox21_file)
print('Reading {}... {} data loaded.'.format(tox21_file, len(tox21_data)))
tox21_data.head()


data = [tox21_data]
data = SMILES2CANONICAL(data)

data_len = len(data[0])
# pick 100 random indices within len(data[0])  = 8014, once leave in/once remove the point from the training data (but put it to the test set)
# for each generate 500 splits (1000 different seeds) with and without the point in the training data/ meaning 1000 total splits

sel_indices = random.sample(range(data_len), 10)
N_splits    = 500
np.save("./splits/sel_indices.npy", np.array([sel_indices, N_splits ]) )

for ind in sel_indices:
    mol_row = [pd.DataFrame(copy.deepcopy(data)[0].loc[ind]).T]
    print(mol_row[0][tox21_tasks[0]])
    check = mol_row[0][tox21_tasks[0]].values[0]
    if check_float_not_nan(check):
        torch.save(mol_row, f"./splits/{ind}.pth")
        for inout in range(2):
            for n_s in range(N_splits):
                data_path = f"./splits/{ind}_{n_s}_{inout}_"
            
                if inout == 0:
                    data_copy = [copy.deepcopy(data)[0].drop(ind, axis=0)]
                    data_copy = apply_splitting(data_copy)
                    save_data(data_copy, data_path)
                
                if inout == 1:
                    data_copy = [copy.deepcopy(data)[0]]
                    data_copy = apply_splitting(data_copy)
                    save_data(data_copy, data_path)
    else:
        print("nan")