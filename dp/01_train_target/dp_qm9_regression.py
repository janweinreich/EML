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

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)




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


class QM9Dataset(Dataset):
    def __init__(self, path_to_file):
        N = 133885
        self.load_X = True
        self.qm9 = np.load(path_to_file, allow_pickle=True)
        self.coords = self.qm9['coordinates']
        self.nuclear_charges = self.qm9['charges']
        self.elements = self.qm9['elements']
        self.energies = np.array(self.qm9['H_atomization'])
        self.Cvs = np.array(self.qm9['Cv'])

        #shuffle all the data and take the first N
        idx = np.arange(len(self.coords))
        np.random.shuffle(idx)
        self.coords = self.coords[idx[:N]]
        self.nuclear_charges = self.nuclear_charges[idx[:N]]
        self.elements = self.elements[idx[:N]]
        self.energies = self.energies[idx[:N]]
        self.Cvs = self.Cvs[idx[:N]]

        if self.load_X:
            self.X = np.load("./tmp/X.npy")
        else:
            self.generate_representations()
            np.save("./tmp/X.npy", self.X)


        

        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            'X': torch.tensor(self.X[idx], dtype=torch.float32),
            'energies': torch.tensor(self.energies[idx], dtype=torch.float32)
        }
        return sample

    
    def generate_representations(self):
        
        print("Generating spahm representations...")
        X = []
        for els, coord in tqdm(zip(self.elements, self.coords), total=len(self.coords)):
            mol_name = mol_to_xyz(els, coord)

            mol = compound.xyz_to_mol(mol_name, 'def2svp', charge=0, spin=0)
            os.remove(mol_name)
            X.append(spahm.compute_spahm.get_spahm_representation(mol, "lb")[0])

        
        self.X = X
        self.X = self.pad_max_size(self.X)


    def pad_max_size(self, X):
        max_size = max([len(subarray) for subarray in X])
        X= np.array([np.pad(subarray, (0, max_size - len(subarray)), 'constant') for subarray in X])
        return X




def mse_loss_numpy(y_true, y_pred):
    mse_loss_numpy = np.mean((y_true - y_pred) ** 2)
    return mse_loss_numpy



class FlexibleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FlexibleNN, self).__init__()
        
        self.hidden_layers = nn.ModuleList()
        
        self.input_layer = nn.Linear(input_size, hidden_size) # input layer
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size)) # hidden layers
        
        self.output_layer = nn.Linear(hidden_size, 1) # output layer for regression
        
    def forward(self, x):
        x = torch.relu(self.input_layer(x)) # pass through input layer
        for layer in self.hidden_layers:
            x = torch.relu(layer(x)) # pass through each hidden layer
        
        x = self.output_layer(x) # pass through output layer
        return x






if __name__ == "__main__":
    DP = False 
    path_to_file = "qm9_data.npz"
    dataset = QM9Dataset(path_to_file)


    # Split the dataset into a training set and a validation set
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 100
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    max_epochs_without_improvement = 5000

    # Model parameters
    input_size = dataset.X.shape[1]  # 35
    hidden_size = 120

    model = FlexibleNN(input_size, hidden_size, 5)

    criterion = nn.MSELoss()


    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    if DP:
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.4,
            max_grad_norm=1.0,
        )

        

    # Training loop 
    num_epochs = 50000
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs = data['X']
            energies = data['energies'].unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, energies)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / (i + 1)}")
        
        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs = data['X']
                energies = data['energies'].unsqueeze(1)

                outputs = model(inputs)
                loss    = criterion(outputs, energies)

                running_val_loss += loss.item()
        
        val_loss = running_val_loss / (i + 1)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if DP:
                torch.save(model.state_dict(), "./tmp/best_model_dp.pt")
            else:
                torch.save(model.state_dict(), "./tmp/best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= max_epochs_without_improvement:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    X_train, X_val =  train_dataset.dataset.X[train_dataset.indices], val_dataset.dataset.X[val_dataset.indices]
    y_train, y_val =  train_dataset.dataset.energies[train_dataset.indices], val_dataset.dataset.energies[val_dataset.indices]

    #save 
    np.savez_compressed("./tmp/dataset.npz", X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)



    model.predict = lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    #predict all test data
    y_hat = []

    for x in tqdm(X_val, desc='Predicting'):
        y_hat.append(model.predict(x.reshape(1, -1))) # Reshape if necessary, as some models expect 2D input.

    # Convert the list to a numpy array to have the same format as before
    y_hat = np.array(y_hat).flatten()

    mae = mean_absolute_error(y_val, y_hat)
    mse_loss = mse_loss_numpy(y_val, y_hat)

    print("MAE: ", mae, mse_loss)
    #exit()

    #plot the results
    fig, ax = plt.subplots()
    ax.plot(y_val, y_hat, alpha=0.5, marker='o', linestyle='')
    #pdb.set_trace()
    #add diagonal line
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_xlabel("True energies")
    ax.set_ylabel("Predicted energies")
    #plt.show()
    plt.savefig("t.png")
