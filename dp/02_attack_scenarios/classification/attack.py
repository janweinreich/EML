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
import numpy as np
import torch
from dp_qm9_classification import FlexibleNN


def batch_sampled_distance(X_guess, X_set, batch_size=1000, num_batches=10):
    total_distance = 0
    total_points = 0
    
    for _ in range(num_batches):
        # Randomly sample a batch from X_guess and X_train
        guess_indices = np.random.choice(len(X_guess), size=batch_size, replace=True)
        train_indices = np.random.choice(len(X_set), size=batch_size, replace=True)

        guess_batch = X_guess[guess_indices]
        train_batch = X_set[train_indices]

        # Compute the distances for the current batch
        differences = guess_batch[:, np.newaxis] - train_batch[np.newaxis, :]
        distances = np.sqrt(np.sum(differences**2, axis=-1))

        total_distance += np.sum(distances)
        total_points += distances.size

    # Compute the average distance
    average_distance = total_distance / total_points
    
    return average_distance




if __name__ == "__main__":
    DP = False
    load_path = '/home/jan/projects/EML/dp/01_train_target/tmp/classification/dataset.npz'
    loaded_arrays = np.load(load_path)
    X_train, X_val, y_train, y_val = np.load(load_path).values()
    X_shadow = np.concatenate((X_train, X_val))

    # Load the model
    input_size = X_train.shape[1]
    hidden_size = 120
    

    model = FlexibleNN(input_size, hidden_size, 5)
    model.load_state_dict(torch.load("/home/jan/projects/EML/dp/01_train_target/tmp/classification/best_model.pt"))

    print("Model loaded")
    model.eval()
    model.predict = lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()



    y_hat = []

    for x in tqdm(X_shadow, desc='Predicting'):
        y_hat.append(model.predict(x)) # Reshape if necessary, as some models expect 2D input.

    y_hat = np.array(y_hat)
    probabilities = nn.functional.softmax(torch.tensor(y_hat), dim=-1).numpy()
    print(probabilities)

    max_vals = np.max(probabilities, axis=1)
    #len(np.argwhere((np.argsort(max_vals)[::-1][:300] < len(X_train))))

    # Define your range and empty score list
    cutrange = range(1, len(X_shadow), 5)
    score, avg_prob = [], []
    avg_rep_distances  = []
    # Assuming max_vals and X_train are predefined
    # study average distance of guessed representations to the training set representations
    for ind, cut in enumerate(cutrange):
        likely_indices = np.argwhere((np.argsort(max_vals)[::-1][:cut] < len(X_train))).flatten()
        guess_score = len(likely_indices)/cut
        avg_prob.append(np.mean(max_vals[np.argsort(max_vals)[::-1][:cut]]))
        score.append(guess_score)

        X_guess = X_shadow[np.argsort(max_vals)[::-1][:cut]]
        #Select random subset of X_train
        X_train_subset = X_train[np.random.choice(len(X_train), size=len(X_guess), replace=True)]
        #Select random subset of X_val
        X_val_subset = X_val[np.random.choice(len(X_val), size=len(X_guess), replace=True)]
        print ( batch_sampled_distance(X_guess, X_train_subset), batch_sampled_distance(X_guess, X_val_subset))






    

    #pdb.set_trace()
    # Create a figure and an axis with a specific size
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Add grid
    ax1.grid(True)

    # Plot the data with a thicker line
    ax1.plot(cutrange, score, linewidth=2)
    ax1.plot(cutrange, avg_prob, linewidth=2)
    
    ax1.axvline(x=len(X_train), color='r', linestyle='--')
    # Add title and labels with a larger font size
    ax1.set_title('Model Evaluation', fontsize=18)
    ax1.set_xlabel('Cutoff (log scale)', fontsize=14)
    ax1.set_ylabel('Score', fontsize=14)

    # Set x-axis to log scale
    ax1.set_xscale('log')

    # Increase the size of the tick labels
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Add a legend
    ax1.legend(['Score'], fontsize=12)

    # Show the plot
    ax1.set_xlim([1e3, max(cutrange)])
    ax1.set_ylim([0.5,np.nanmax(avg_prob)])

    plt.savefig('score.png')
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(cutrange, avg_rep_distances, linewidth=2)
    ax2.set_title('Average distance of guessed representations to the training set representations', fontsize=18)
    ax2.set_xlabel('Cutoff (log scale)', fontsize=14)
    ax2.set_ylabel('Average distance', fontsize=14)
    ax2.set_xscale('log')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(['Average distance'], fontsize=12)
    ax2.set_xlim([1e3, max(cutrange)])
    plt.savefig('avg_rep_distances.png')
    print(avg_rep_distances)
    plt.close()
