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
from sklearn.model_selection import train_test_split
#import mean absolute error
from sklearn.metrics import mean_absolute_error
import random
import matplotlib.pyplot as plt

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
        N = 10000 #133885
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

        if self.load_X == True:
            self.X = np.load("./tmp/X.npy")
        else:
            self.generate_representations()
            np.save("./tmp/X.npy", self.X)
        # train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.energies, test_size=0.2, random_state=42)
        print("Train size: ", len(self.X_train))
        print("Test size: ", len(self.X_test))


    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        sample = {
            'X': torch.tensor(self.X_train[idx], dtype=torch.float32),
            'energies': torch.tensor(self.y_train[idx], dtype=torch.float32)
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


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)  # Single scalar output for regression

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def eval_test(model, X_test, y_test):
    y_pred = model(torch.tensor(X_test, dtype=torch.float32))
    y_pred = y_pred.detach().numpy()
    mae = mean_absolute_error(y_test, y_pred)
    return mae


if __name__ == "__main__":
    DP = True 
    path_to_file = "qm9_data.npz"
    dataset = QM9Dataset(path_to_file)
    # Create the DataLoader
    batch_size = 100
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



    input_size = dataset.X.shape[1]  # 35
    hidden_size1 = 64
    hidden_size2 = 32
    model = SimpleNN(input_size, hidden_size1, hidden_size2)

    criterion = nn.MSELoss()


    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    if DP:
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )

        

    # Training loop 
    num_epochs = 120
    for epoch in range(num_epochs):
        
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs = data['X']
            energies = data['energies'].unsqueeze(1)  # Add an extra dimension to match the model's output

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, energies)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss per epoch
        print(f"Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}")
        print("Epoch: ", epoch , "test loss: ", eval_test(model, dataset.X_test, dataset.y_test))
        #pdb.set_trace()
        
        

    model.predict = lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    #predict all test data
    y_pred = model.predict(dataset.X_test)
    #pdb.set_trace()
    #eval_test
    print("MAE: ", eval_test(model, dataset.X_test, dataset.y_test))

    #plot the results
    fig, ax = plt.subplots()
    ax.plot(dataset.y_test, y_pred, alpha=0.5, marker='o', linestyle='')
    #add diagonal line
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_xlabel("True energies")
    ax.set_ylabel("Predicted energies")
    plt.show()

    #effect of normalization?