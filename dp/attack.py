import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from qstack import compound, spahm
import os
from tqdm import tqdm
import pickle
import pdb
import torch
import torch.nn as nn
from opacus import PrivacyEngine
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import random
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import copy
import random
from dp_qm9 import SimpleNN

def estimate_typical_distance(synthetic_data):
    distances = []
    for i in range(len(synthetic_data) - 1):
        for j in range(i+1, len(synthetic_data)):
            distances.append(np.linalg.norm(synthetic_data[i] - synthetic_data[j]))
    return np.median(distances)

def random_perturbation(x, typical_distance):
    return x + np.random.normal(loc=0.0, scale=typical_distance, size=x.shape)

def sensitivity(model, x, typical_distance, n_perturbations=100):
    sensitivities = []
    for _ in range(n_perturbations):
        x_perturbed = random_perturbation(x, typical_distance)
        x_pred = model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        x_perturbed_pred = model(torch.tensor(x_perturbed, dtype=torch.float32)).detach().numpy()
        sensitivities.append(np.abs(x_perturbed_pred - x_pred))
    return np.mean(sensitivities)

def get_sensitivity(model, x_hack, n_perturbations=100, sample_size=100):
    model.eval()

    typical_distance = 0.1  #estimate_typical_distance(synthetic_data)

    # Calculate the model's sensitivity to x_hack
    x_hack_sensitivity = sensitivity(model, x_hack, typical_distance, n_perturbations)

    return x_hack_sensitivity


if __name__ == "__main__":
    load_path = './tmp/dataset.npz'
    loaded_arrays = np.load(load_path)
    X_train, X_val, y_train, y_val = np.load(load_path).values()
    X_shadow = np.concatenate((X_train, X_val))
    y_shadow = np.concatenate((y_train, y_val))

    # Load the model
    
    input_size = X_train.shape[1]  # 35
    hidden_size1 = 64
    hidden_size2 = 64


    model = SimpleNN(input_size, hidden_size1, hidden_size2)
    model.load_state_dict(torch.load("./tmp/best_model.pt"))

    model_DP = copy.deepcopy(model)
    state_dict = torch.load("./tmp/best_model_dp.pt")
    new_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
    model_DP.load_state_dict(new_state_dict)


    #for all inital shadow data, calculate the sensitivity
    results     = np.array([get_sensitivity(model, X_shadow[i]) for i in tqdm(range(len(X_shadow)))])
    #look at distrubtion of sensitivities
    #find that
    #all points with sensitivity > 0.025 are in the test set as one would expect
    theshold    = np.mean(results) + np.std(results)
    print(len(np.where(np.argwhere(results > 0.025).flatten() > 8000)))
    #indicating that the first 8000 points are in the training set
    #and the rest are in the test set


    


    #0.025
    plt.show()
    print(results)
    #dp_results  = inversion_attack(model_DP, X_train[666], X_val)


    exit()
    # Plot the results, TWO PLOTS
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(results[1], bins=20, alpha=0.5, label='Synthetic data')
    axs[0].axvline(results[0], color='r', linestyle='dashed', linewidth=1, label='x_hack')
    #plot mean of synthetic sensitivities as well
    axs[0].axvline(np.mean(results[1]), color='g', linestyle='dashed', linewidth=1, label='mean of synthetic sensitivities')
    axs[0].set_title('Non-DP model')
    axs[0].set_xlabel('Sensitivity')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()
    #same for dp model
    axs[1].hist(dp_results[1], bins=20, alpha=0.5, label='Synthetic data')
    axs[1].axvline(dp_results[0], color='r', linestyle='dashed', linewidth=1, label='x_hack')
    #plot mean of synthetic sensitivities as well
    axs[1].axvline(np.mean(dp_results[1]), color='g', linestyle='dashed', linewidth=1, label='mean of synthetic sensitivities')
    axs[1].set_title('DP model')
    axs[1].set_xlabel('Sensitivity')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    plt.tight_layout()
    plt.savefig('./tmp/attack_non_dp.png')