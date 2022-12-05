# EML
Encrypted machine learning of molecular quantum properties

Large machine learning (ML) models with improved predictions have become widely available in the chemical sciences. 
Unfortunately, these models do not protect the privacy necessary within commercial settings, prohibiting the use of potentially extremely valuable data by others. Encrypting the prediction process can solve this problem by double-blinded model evaluation and prohibits the extraction of training or query data.
Unfortunately, contemporary ML models based on fully homomorphic encryption or federated learning are either too expensive for practical use or have to trade higher speed for weaker security. 

We have implemented secure and efficient encrypted machine learning (EML) models using oblivious transfer enabling efficient and secure predictions of molecular quantum properties across chemical compound space. The performance of our encrypted kernel-ridge regression models indicates a dire need for a compact ML model architecture, including molecular representation and kernel matrix size, that minimizes model evaluation costs.

# Encypted Kernel-Ridge Regression Predictions


# Encrypted Neural Network Predictions:

AliceNet: This is the name of the class that represents an encrypted neural network.

nn.Module: This is the parent class from PyTorch that AliceNet extends. nn.Module provides a base class for all neural network modules in PyTorch.

__init__: This is the constructor of the AliceNet class. It takes several parameters to initialize the network, such as the input dimension, number of neurons in each layer, learning rate, and number of epochs.

forward: This is a method of the AliceNet class that specifies the forward pass of the network. It takes an input tensor x and applies a sequence of linear transformations and ReLU activation functions to compute the output of the network.

fit: This is a method of the AliceNet class that trains the network on encrypted data using mini-batch training. It takes as input the training data X_train and labels y_train, and uses the Adam optimizer to minimize the mean squared error loss. The method trains the network for the number of epochs specified in the constructor, and updates the network weights after each mini-batch.

fc1, fc2, ..., fc9: These are the linear transformation layers of the network. The fc1 layer has in_dim input units and 2*n_neuro output units, and the remaining layers have n_neuro input and output units. These layers are fully connected, meaning that each unit in a layer is connected to all units in the previous and next layers.

parameters: This is a method of the nn.Module class that returns an iterator over the parameters of the network. It is used by the Adam optimizer to update the network weights during training.

lr: This is the learning rate of the network, which determines the size of the weight updates during training. It is a hyperparameter that is set to a fixed value in the constructor.

num_epochs: This is the number of epochs to train the network for. It is a hyperparameter that is set to a fixed value in the constructor.

loss: This is an instance of the mean squared error loss function from PyTorch. It is used to compute the difference between the predicted output of the network and the true labels during training.

optimizer: This is an instance of the Adam optimizer from PyTorch. It is used to update the network weights during training in order to minimize the loss.

batch_size: This is the number of samples to include in each mini-batch during training. It is a hyperparameter that is set to a fixed value in the fit method.

# Setup

1) Download the OT software package MP-SPDZ (tested with version mp-spdz-0.2.5) from https://github.com/data61/MP-SPDZ

