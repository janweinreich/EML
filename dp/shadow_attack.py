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
from dp_qm9 import FlexibleNN
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report




class Target_model:
    def __init__(self, dp_model = False) -> None:
        self.dp_model = dp_model
        self.load_target_model()

    def load_target_model(self):
        input_size = X_train.shape[1]
        hidden_size = 120
        self.model = FlexibleNN(input_size, hidden_size, 5)
        if self.dp_model:
            self.model.load_state_dict(torch.load("./tmp/best_model_dp.pt"))
        else:
            self.model.load_state_dict(torch.load("./tmp/best_model.pt"))

    def predict(self, X):
        self.model.eval()
        y_hat = self.model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        return y_hat

class Shadow_data(Dataset):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)



    def __getitem__(self, idx):
        sample = {
            'X': torch.tensor(self.X[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx], dtype=torch.float32)
        }
        return sample




class Shadow_models:
    def __init__(self,X_shadow, y_shadow, k=2) -> None:
        self.X_shadow = X_shadow
        self.y_shadow = y_shadow

        self.batch_size = 100
        self.k = k
        self.shadow_models = []
        self.train_models()

    def train_models(self):
        kfold = KFold(n_splits=self.k, shuffle=True, random_state=42)
        k = 0

        self.train_split_shadows = []
        for train_index, test_index in kfold.split(self.X_shadow):
            X_train, X_test = self.X_shadow[train_index], self.X_shadow[test_index]
            y_train, y_test = self.y_shadow[train_index], self.y_shadow[test_index]

            self.train_split_shadows.append([X_train, X_test, y_train, y_test])
            train_dataset, val_dataset = Shadow_data(X=X_train, y=y_train), Shadow_data(X=X_test, y=y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            curr_shadow = self.train_single_model(train_dataset, train_loader, val_dataset, val_loader, k)
            self.shadow_models.append(curr_shadow)

            k+=1

    def evaluate_shadows(self):

        #for all (x, y) ∈ Dtrain create (y, y, in) 
        #for all (x, y) ∈ Dtest create (y, y, out)

        X_fodder, y_fodder = [], []
        for curr_shadow, split_data in zip(self.shadow_models, self.train_split_shadows):
            
            X_train, X_test, y_train, y_test = split_data
            y_hat_train_shadow = curr_shadow(torch.tensor(X_train, dtype=torch.float32)).detach().numpy()
            y_hat_test_shadow = curr_shadow(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
            curr_x_fodder_train = np.stack((y_train.flatten(), y_hat_train_shadow.flatten()), axis=1)
            curr_y_fodder_train = np.ones_like(np.arange(len(curr_x_fodder_train)))
            curr_x_fodder_test = np.stack((y_test.flatten(), y_hat_test_shadow.flatten()), axis=1)
            curr_y_fodder_test = np.zeros_like(np.arange(len(curr_x_fodder_test)))

            X_fodder.append(np.concatenate((curr_x_fodder_train, curr_x_fodder_test)))
            y_fodder.append(np.concatenate((curr_y_fodder_train, curr_y_fodder_test)))

        self.X_fodder, self.y_fodder = np.concatenate(X_fodder), np.concatenate(y_fodder)


    def train_attack_model(self):
        # First, split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X_fodder, self.y_fodder, test_size=0.2, random_state=42)

        # Initialize the classifier
        classifier = LogisticRegression()

        # Train the classifier
        classifier.fit(X_train, y_train)

        # Use the trained classifier to make predictions on the test set
        y_pred = classifier.predict(X_test)

        # Display the classification report
        print(classification_report(y_test, y_pred))



    def train_single_model(self, train_dataset, train_loader, val_dataset, val_loader, k):
        # Define training constants
        max_epochs_without_improvement = 100
        num_epochs = 1000 # 10000
        learning_rate = 0.01
        input_size = train_dataset.X.shape[1]
        hidden_size = 120

        # Initialize the model and loss function
        model = FlexibleNN(input_size, hidden_size, 5)
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Define the best validation loss and counter for epochs without improvement
        best_val_loss = float('inf')
        epochs_without_improvement = 100

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            training_loss = self.train_epoch(train_loader, model, loss_function, optimizer)
            print(f"Epoch {epoch + 1}, Training Loss: {training_loss}")
            
            model.eval()
            validation_loss = self.validate_epoch(val_loader, model, loss_function)
            print(f"Epoch {epoch + 1}, Validation Loss: {validation_loss}")

            # Check for early stopping
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f"./tmp/shadow_models/shadow_{k}.pt")
               
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= max_epochs_without_improvement:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return model

    def train_epoch(self, loader, model, loss_function, optimizer):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs = data ['X']
            targets = data['y'].unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        return running_loss / (i + 1)

    def validate_epoch(self, loader, model, loss_function):
        running_val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(loader, 0):
                inputs = data['X']
                targets = data['y'].unsqueeze(1)

                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                running_val_loss += loss.item()
        return running_val_loss / (i + 1)



if __name__ == "__main__":
    load_path = './tmp/dataset.npz'
    loaded_arrays = np.load(load_path)
    X_train, X_val, y_train, y_val = np.load(load_path).values()
    X_shadow    = np.concatenate((X_train, X_val))
    y_true      = np.concatenate((y_train, y_val))
    y_shadow    = Target_model(dp_model=False).predict(X_shadow)


    shadow_class = Shadow_models(X_shadow, y_shadow, k=2)
    shadow_class.evaluate_shadows()
    shadow_class.train_attack_model()