import torch
import torch.nn as nn 
import torch.nn as nn
import torch.nn.functional as F
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import random
from tqdm import tqdm
random.seed(1337)
np.random.seed(1337)


class AliceNet(nn.Module):
    """
    Class for an encrypted neural network
    """

    def __init__(self, in_dim=None, n_neuro=None, lr=None, num_epochs=None,X_test=None, y_test=None, name=None, verbose=None):
        """
        in_dim: input dimension
        n_neuro: number of neurons in each layer
        lr: learning rate
        num_epochs: number of epochs
        X_test: test data
        y_test: test labels
        name: name of the model
        verbose: print loss every 100 epochs
        """
        super(AliceNet, self).__init__()

        self.in_dim      = in_dim or 276
        self.n_neuo      = n_neuro or 300
        self.lr          = lr or 0.0003
        self.num_epochs  = num_epochs or 10
        self.name        = name or "ALICE_NET"
        self.y_test      = y_test #or None
        self.X_test      = X_test
        self.loss        = nn.MSELoss()
        self.verbose     = verbose or False
        
        self.fc1 = nn.Linear(self.in_dim, 2*self.n_neuo)
        self.fc2 = nn.Linear(2*self.n_neuo, self.n_neuo)
        self.fc3 = nn.Linear(self.n_neuo, self.n_neuo)
        self.fc4 = nn.Linear(self.n_neuo, self.n_neuo)
        self.fc5 = nn.Linear(self.n_neuo, self.n_neuo)
        self.fc6 = nn.Linear(self.n_neuo, self.n_neuo)
        self.fc7 = nn.Linear(self.n_neuo, self.n_neuo)
        self.fc8 = nn.Linear(self.n_neuo, self.n_neuo)
        self.fc9 = nn.Linear(self.n_neuo, 1)
 
    def forward(self, x):
        """
        forward pass
        """
        act = F.relu #selu works much much better
        out = self.fc1(x)
        out = act(out)
        out = self.fc2(out)
        out = act(out)
        out = self.fc3(out)
        out = act(out)
        out = self.fc4(out)
        out = act(out)
        out = self.fc5(out)
        out = act(out)
        out = self.fc6(out)
        out = act(out)
        out = self.fc7(out)
        out = act(out)
        out = self.fc8(out)
        out = act(out)
        out = self.fc9(out)
        return out


    def fit(self, X_train, y_train):
        """
        fit the model to the data X_train, y_train
        minibatch training is used to train the model on the encrypted data
        """
        optimizer = torch.optim.Adam(self.parameters(), lr =self.lr )
        batch_size = 256 # or whatever

        for ep in range(self.num_epochs):

            permutation = torch.randperm(X_train.size()[0])

            for i in range(0,X_train.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
                y_pred = model.forward(batch_x)
            
                loss= self.loss(y_pred, batch_y)

                loss.backward()
                optimizer.step()
            if self.verbose and ep%100==0:
                print('epoch {}, loss {}'.format(ep, loss.item()))

            valid_loss = self.loss(self.y_test, self.forward(self.X_test))
            if loss < valid_loss*0.0015:
                break


    def predict(self, X_test):
        """
        predict the labels of the test data
        """
        return self(X_test)

    def save(self):
        """
        save the model
        """
        torch.save(self, self.name)


class Crypto():
    """
    Class for the crypto network,
    handles the communication between the two parties
    """

    global ALICE
    global BOB
    ALICE, BOB = 0,1

    def __init__(self, model_saved, BOB_DATA):
        self.model      = torch.load(model_saved)
        self.bob_data   = BOB_DATA
        
    @mpc.run_multiprocess(world_size=2)
    def encrypt_query_data(self):
        crypten.save_from_party(self.bob_data,"BOB_data", src=BOB)

    def encrypt_model(self):
        crypten.init()
        torch.set_num_threads(1)
        crypten.common.serial.register_safe_class(AliceNet)

        first_parameter = next(self.model.parameters())
        input_shape = first_parameter.size()[1]

        dummy_input = torch.empty((1, input_shape))
        self.private_model = crypten.nn.from_pytorch(self.model, dummy_input)
        self.private_model.encrypt(src=ALICE)

    @mpc.run_multiprocess(world_size=2)
    def encrypted_pred(self):
        self.encrypt_model()
        self.encrypt_query_data()

        self.query_data =  crypten.load_from_party("BOB_data",  src=BOB)
        self.private_model.eval()
        output_enc = self.private_model(self.query_data)
        output = output_enc.get_plain_text()
        output = np.array(output.detach().numpy())
        return output


class Scaler():
    def __init__(self):
        self.maxval = None
        self.minval = None

    def fit(self, X):
        self.maxval  = np.max(X, axis=0)
        self.minval  = np.min(X, axis=0)
    
    
    def transform(self,X):
        X_std = (X - self.minval / (self.maxval - self.minval))
        X_scaled = X_std * (self.maxval- self.minval) + self.minval
        return X_scaled



if __name__ == "__main__":
    fit = False

    if fit:

        new = False

        if new:
            import get_qm9 
            #load the complete QM9 dataset in BoB representation and atomization energies
            X, y = get_qm9.get_qm9(140000)

            np.savez_compressed("data",X=X, y=y)
        else:
            data = np.load("data.npz", allow_pickle=True)
            X, y = data["X"], data["y"]


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        y_train = y_train+ np.random.normal(0, 1e-4*np.std(y_train), y_train.shape)

        y_train, y_test     = y_train.reshape(-1,1), y_test.reshape(-1,1)



        #scale the data to the range [0,1] for better training performance of the neural network
        sc          = MinMaxScaler()
        sct         = MinMaxScaler()
        X_train     = sc.fit_transform(X_train)
        X_test      = sc.transform(X_test)
        y_train     = sct.fit_transform(y_train)
        y_test      = sct.transform(y_test)

        print(len(y_train))
        print(len(y_test))


        #convert the data to torch tensors
        X_train     = torch.from_numpy(X_train.astype(np.float32))
        X_test      = torch.from_numpy(X_test.astype(np.float32))
        y_train     = torch.from_numpy(y_train.astype(np.float32))
        y_test      = torch.from_numpy(y_test.astype(np.float32))



        N = [2**i for i in range(6,17)]
        N.append(len(y_train))
        for n in N:
            model = AliceNet(in_dim=X_train.shape[1], n_neuro=250, num_epochs=8000,y_test=y_test.cuda(), X_test=X_test.cuda(),verbose=True)
            model.to(torch.device("cuda"))
            model.fit(X_train[:n].cuda(), y_train[:n].cuda())
            model.to(torch.device("cpu"))
            model.save()


            crypto_predictions    = []
            plaintext_predictions = []
            
            #predict using the encrypted model and without encryption
            #need to rescacle the data to the original range after prediction


            for i in tqdm(range(len(X_test))       ) :
                X_curr = X_test[i].reshape((1, X_test.shape[1]))
                qml_crypto = Crypto("ALICE_NET",X_curr)
                tic = time.perf_counter()
                pred_crypto = sct.inverse_transform( qml_crypto.encrypted_pred()[1]).flatten()
                toc = time.perf_counter()
                dT   = toc -tic

                pred        = sct.inverse_transform(np.array(model.predict(X_curr).detach()))

                crypto_predictions.append(pred_crypto)
                plaintext_predictions.append(pred)

            crypto_predictions, plaintext_predictions  = np.array(crypto_predictions),       np.array(plaintext_predictions)
            crypto_predictions, plaintext_predictions  = np.concatenate(crypto_predictions), np.concatenate(plaintext_predictions).flatten()
            num_mae     = np.mean(np.abs(crypto_predictions-plaintext_predictions))
            mae_plain   = np.mean(np.abs(plaintext_predictions -    sct.inverse_transform(y_test).flatten()))
            mae         = np.mean(np.abs(crypto_predictions    -    sct.inverse_transform(y_test).flatten()))
            print("#",n ,mae_plain, mae,num_mae)
            plt.plot(crypto_predictions, sct.inverse_transform(y_test).flatten(), "o")

