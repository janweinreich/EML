# EML
Encrypted machine learning of molecular quantum properties

Large machine learning (ML) models with improved predictions have become widely available in the chemical sciences. 
Unfortunately, these models do not protect the privacy necessary within commercial settings, prohibiting the use of potentially extremely valuable data by others. Encrypting the prediction process can solve this problem by double-blinded model evaluation and prohibits the extraction of training or query data.
Unfortunately, contemporary ML models based on fully homomorphic encryption or federated learning are either too expensive for practical use or have to trade higher speed for weaker security. 

We have implemented secure and efficient encrypted machine learning (EML) models using oblivious transfer enabling efficient and secure predictions of molecular quantum properties across chemical compound space. The performance of our encrypted kernel-ridge regression models indicates a dire need for a compact ML model architecture, including molecular representation and kernel matrix size, that minimizes model evaluation costs.

# Encypted Kernel-Ridge Regression Predictions

The `CM.mpc` script first sets the numerical precision of its calculations to 42 bits and sets the precision of the output to the terminal. It then defines a few variables, including sigma, which is the width of the kernel used in the regression.

Next, the script reads in training data and test data from two input files, and computes the kernel matrix overlap between the test data and the training data. This is done using a gaussian kernel function, which is applied to the dot product of the test data and the training data.

Finally, the script makes a prediction based on the computed kernel matrix and the training data, and outputs the result to the terminal. 


To submit mutliple jobs use
the `sub.sh` script:

The first loop iterates over the values 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, and 16384, assigning each value in turn to the variable ntrain. The second loop iterates over the values 0 to 19, incrementing by 1 each time, and assigns each value to the variable ntest.

For each combination of ntrain and ntest, the script performs the following operations:

Copies the file main_sub.sh to a new file main_sub_{ntrain}_{ntest}.sh, where ${ntrain} and ${ntest} are the values of the ntrain and ntest variables, respectively.
Replaces the string reptype= with the value of the reptype variable in the newly-created file main_sub_{ntrain}_{ntest}.sh.
Replaces the string #SBATCH --job-name= with the string #SBATCH --job-name={reptype}_${ntrain}_${ntest} in the newly-created file main_sub_${ntrain}_${ntest}.sh, where {reptype}, {ntrain}, and ${ntest} are the values of the reptype, ntrain, and ntest variables, respectively.
Replaces the string ntrain= with the value of the ntrain variable in the newly-created file main_sub_{ntrain}_{ntest}.sh.
Replaces the string ntest= with the value of the ntest variable in the newly-created file main_sub_{ntrain}_{ntest}.sh.
Submits the newly-created file main_sub_${ntrain}_${ntest}.sh for execution on a computer cluster using the sbatch command.
Moves the newly-created file main_sub_{ntrain}_{ntest}.sh to the directory ./jobscripts/{reptype}, where {reptype} is the value of the reptype variable.



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

