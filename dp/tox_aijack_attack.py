from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aijack.attack.membership import ShadowMembershipInferenceAttack
from aijack.utils.utils import TorchClassifier, NumpyDataset
import numpy as np
import pandas as pd
from scipy import io
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pdb
import pickle
from opacus import PrivacyEngine
from torch.utils.data import random_split
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

#Source of the dataset:
#http://bioinf.jku.at/research/DeepTox/tox21.html
#http://bioinf.jku.at/research/DeepTox/sampleCode.py 
#how ti lead the dataset


def dump_to_file(obj, file_path):
    """Dump a Python object to a file using pickle."""
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_from_file(file_path):
    """Load a Python object from a file using pickle."""
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj



class Tox21Dataset:
    def __init__(self) -> None:
        data = load_from_file("./archive/tox21_NR.AhR.pkl")
        self.X, self.y = data[0], data[1]

    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        sample = {
            'X': torch.tensor(self.X[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx], dtype=torch.float32)
        }
        return sample



y_tr = pd.read_csv('./archive/tox21_labels_train.csv.gz', index_col=0, compression="gzip")
y_te = pd.read_csv('./archive/tox21_labels_test.csv.gz', index_col=0, compression="gzip")
x_tr_dense = pd.read_csv('./archive/tox21_dense_train.csv.gz', index_col=0, compression="gzip").values
x_te_dense = pd.read_csv('./archive/tox21_dense_test.csv.gz', index_col=0, compression="gzip").values
x_tr_sparse = io.mmread('./archive/tox21_sparse_train.mtx.gz').tocsc()
x_te_sparse = io.mmread('./archive/tox21_sparse_test.mtx.gz').tocsc()

# filter out very sparse features
sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()
x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].A])
x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].A])


target = "NR.AhR"
rows_tr = np.isfinite(y_tr[target]).values
rows_te = np.isfinite(y_te[target]).values

X, y = x_tr[rows_tr], np.int_(y_tr[target][rows_tr].values)


max_N = 4000

X, y = X[:max_N], y[:max_N]


# We use the train dataset to train the victim model. The attacker utilize shadow dataset to
# prepare membership inference attack. The test dataset is used to evaluate the result of attack.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=42
)
X_train, X_shadow, y_train, y_shadow = train_test_split(
    X_train, y_train, test_size=1 / 2, random_state=42
)
# We simulate the situation where the distribution of training dataset is different from the test/shadow datasets.
X_test = 0.5 * X_test + 0.5 * np.random.normal(size=(X_test.shape))



clf = SVC(probability=True)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train), clf.score(X_test, y_test))


# Train the attacker

shadow_models = [SVC(probability=True) for _ in range(5)]
attack_models = [SVC(probability=True) for _ in range(15)]

attacker = ShadowMembershipInferenceAttack(clf, shadow_models, attack_models)
attacker.fit(X_shadow, y_shadow)


# Get the attack result of membership inference
in_result = attacker.predict(clf.predict_proba(X_train), y_train)
out_result = attacker.predict(clf.predict_proba(X_test), y_test)

in_label = np.ones(in_result.shape[0])
out_label = np.zeros(out_result.shape[0])

score_SVM = accuracy_score(np.concatenate([in_label, out_label]), np.concatenate([in_result, out_result]))
print("Accuracy of membership inference attack: {}".format(score_SVM))

print("Now same Spiel with neural networks")

class LM(nn.Module):
    def __init__(self):
        super(LM, self).__init__()
        self.lin1 = nn.Linear(1644, 2)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(2, 2)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(2, 2)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(2, 2)
        self.relu4 = nn.ReLU()
        self.lin5 = nn.Linear(2, 1)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu1(out)
        out = self.lin2(out)
        out = self.relu2(out)
        out = self.lin3(out)
        out = self.relu3(out)
        out = self.lin4(out)
        out = self.relu4(out)
        out = self.lin5(out)
        return torch.sigmoid(out)
    
# Train the victim
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

criterion = nn.BCELoss()
net = LM().to(torch.double) #.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01)
dataset = Tox21Dataset()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Early stopping parameters
best_val_loss = float('inf')
epochs_without_improvement = 0
max_epochs_without_improvement = 10

DP = False
if DP:
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.5,
        max_grad_norm=1.0,
    )
else:
    model = net
    optimizer = optimizer

# Training loop 
num_epochs = 10000
for epoch in range(num_epochs):
    # Training loop
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs = data['X'].double()
        y = data['y'].double().unsqueeze(1)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Training Loss: {running_loss / (i + 1)}")
    
    # Validation loop
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs = data['X'].double()
            y = data['y'].double().unsqueeze(1)

            outputs = model(inputs)
            loss    = criterion(outputs, y)
            #pdb.set_trace()
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
y_train, y_val =  train_dataset.dataset.y[train_dataset.indices], val_dataset.dataset.y[val_dataset.indices]

# Ensure your model is in evaluation mode
model = model.float()
model.eval()

# Convert your data to a PyTorch tensor and ensure it's on the correct device
X_val_tensor = torch.from_numpy(X_val).float().to(device)

# Use your model to make predictions
with torch.no_grad():
    predictions = model(X_val_tensor)

# If you want to convert these predictions to a numpy array:
y_hat = predictions.cpu().numpy().flatten()

plt.plot(y_hat, y_val, 'o')
plt.show()
pdb.set_trace()
# You need to wrap the torch module with TorchClassifier
#clf = TorchClassifier(
#    model, criterion, optimizer, batch_size=64, epoch=100, device=device
#)

#pdb.set_trace()
#clf.fit(X_train, y_train)
#print(clf.score(X_train, y_train), clf.score(X_test, y_test))


def create_clf():
    _net = LM().to(torch.double).to(device)
    _optimizer = optim.Adam(_net.parameters(), lr=0.001)
    return TorchClassifier(
        _net, criterion, _optimizer, batch_size=100, epoch=100, device=device
    )


shadow_models = [create_clf() for _ in range(4)]
attack_models = [SVC(probability=True) for _ in range(14)]

attacker = ShadowMembershipInferenceAttack(clf, shadow_models, attack_models)

#get shadow data not from svm scipt above but from the dataset


y_shadow = y_shadow.astype('float64').reshape(-1, 1)

attacker.fit(X_shadow, y_shadow)


# Get the attack result of membership inference
in_result = attacker.predict(clf.predict_proba(X_train), y_train)
out_result = attacker.predict(clf.predict_proba(X_test), y_test)

in_label = np.ones(in_result.shape[0])
out_label = np.zeros(out_result.shape[0])


score_NN  = accuracy_score(
    np.concatenate([in_label, out_label]), np.concatenate([in_result, out_result])
)
print("Accuracy of membership inference attack: {}".format(score_NN))