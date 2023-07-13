#!/usr/bin/env python
# coding: utf-8

# ##### This notebook creates STDNN-FP (pytorch) for classification prediction on Tox21
# 
# Using split data already saved.
# 
# Notebook shows results for seed = 124, but we also ran on seed 122, 123. 
# 
# Before use define desired pathways to save models,:
# - path variable, in "Create checkpoint" section for models
# - writer variable, in "Train the neural network model" section for tensorboard summary
import numpy as np
import pandas as pd
import os

# Required RDKit modules
import rdkit as rd
from rdkit import DataStructs
from rdkit.Chem import AllChem

# modeling
import sklearn as sk
from sklearn.model_selection import train_test_split

# Graphing
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
import datetime, os
import pdb
from torch.utils.tensorboard import SummaryWriter
import shutil

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, input_model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    input_model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    train_loss_min = checkpoint['train_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], train_loss_min.item()

def Rdkit2Numpy(moldata):
    # convert the RDKit explicit vectors into numpy arrays
    x = []
    for fp in moldata['morgan']:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        x.append(arr)
    x = np.array(x)
    x = x- 0.5
    return x

# DNN Model class
# Each 12 tasks are treated as separate tasks with 2 separate layers
class DNN(torch.nn.Module):
    def __init__(self, input_shape, all_tasks):
        super(DNN, self).__init__()

        self.hidden_1 = torch.nn.ModuleList([torch.nn.Linear(input_shape, 1024) for task in all_tasks])
        self.batchnorm_1 = torch.nn.ModuleList([torch.nn.BatchNorm1d(1024) for task in all_tasks])
        
        self.hidden_2 = torch.nn.ModuleList([torch.nn.Linear(1024, 512) for task in all_tasks])
        self.batchnorm_2 = torch.nn.ModuleList([torch.nn.BatchNorm1d(512) for task in all_tasks])
        
        self.output   = torch.nn.ModuleList([torch.nn.Linear(512, 1) for task in all_tasks])
        
        # function for leaky ReLU
        self.leakyReLU = torch.nn.LeakyReLU(0.05)

    def forward(self, x):        
        x_task = [None for i in range(len(self.output))]  # initialize
        for task in range(len(self.output)):
            x_task[task] = self.hidden_1[task](x)
            x_task[task] = self.batchnorm_1[task](x_task[task])
            x_task[task] = self.leakyReLU(x_task[task])
            
            x_task[task] = self.hidden_2[task](x_task[task])
            x_task[task] = self.batchnorm_2[task](x_task[task])
            x_task[task] = self.leakyReLU(x_task[task])
            
            x_task[task] = self.output[task](x_task[task])
            x_task[task] = torch.sigmoid(x_task[task])
        
        y_pred = x_task
        
        return y_pred

# Class for DNN data
class DNNData(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



def get_rdkit(data):
    
    for i in range(len(data)):
        data[i]['mol'] = [rd.Chem.MolFromSmiles(x) for x in data[i]['smiles']]

        bi = [{} for _ in range(len(data[i]))]
        data[i]['morgan'] = [AllChem.GetMorganFingerprintAsBitVect(data[i].iloc[j]['mol'], morgan_radius, nBits = morgan_bits, bitInfo=bi[j]) 
                            for j in range(len(data[i]))]
        data[i]['bitInfo'] = bi
    
    return data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  


morgan_bits,morgan_radius = 4096, 2
train_epoch = 60 
batch = 512 
tox21_file  = 'tox21.csv'
tox21_tasks = ['SR-ARE']

tox21_data = pd.read_csv(tox21_file)
print('Reading {}... {} data loaded.'.format(tox21_file, len(tox21_data)))
tox21_data.head()

data = [tox21_data] 

all_tasks = [tox21_tasks[0]]

#list all files in the directory ./splits

split_info = np.load('./splits/sel_indices.npy', allow_pickle=True)
sel_indices,N_splits = split_info[0], split_info[1]


for ind in sel_indices:
    mol_row = torch.load(f"./splits/{ind}.pth")
    mol_row = get_rdkit(mol_row)
    mol_row[0] = mol_row[0].fillna(-1)
    x_row,y_row = Rdkit2Numpy(mol_row[0]), mol_row[0][all_tasks].values
    x_row_torch = x_row.astype(np.float32)
    y_row_torch = y_row.astype(np.float32)
    pdb.set_trace()
    for n_s in range(N_splits):
        for inout in range(2):
            seed_value = random.randint(0, 100000)
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed(seed_value)
            np.random.seed(seed_value)
            random.seed(seed_value)

            # load saved tox21 train/test/valid data 
            data_path  = f"./splits/{ind}_{n_s}_{inout}_"
            train_data = torch.load(data_path + 'train.pth')
            test_data  = torch.load(data_path + 'test.pth')
            valid_data = torch.load(data_path + 'valid.pth')

            data = [train_data, test_data, valid_data]
            

            # construct morgan fingerprints 
            data = get_rdkit(data)
            data[0] = data[0].fillna(-1)
            data[1] = data[1].fillna(-1)
            data[2] = data[2].fillna(-1)

            train_data = data[0]
            test_data  = data[1]
            valid_data = data[2]

            
            x_train,x_test, x_valid  = Rdkit2Numpy(train_data),Rdkit2Numpy(test_data), Rdkit2Numpy(valid_data)
            y_train,y_test,y_valid  = train_data[all_tasks].values,test_data[all_tasks].values, valid_data[all_tasks].values


            # count the number of data points per class
            N_train = np.sum(y_train >= 0, 0)
            N_test  = np.sum(y_test >= 0, 0)
            N_valid  = np.sum(y_valid >= 0, 0)

            # convert data for pytorch
            x_train_torch = x_train.astype(np.float32)
            y_train_torch = y_train.astype(np.float32)

            x_test_torch = x_test.astype(np.float32)
            y_test_torch = y_test.astype(np.float32)

            x_valid_torch = x_valid.astype(np.float32)
            y_valid_torch = y_valid.astype(np.float32)
            input_shape = x_train_torch.shape[1]


            training_set = DNNData(x_train_torch, y_train_torch)
            training_generator = DataLoader(training_set, batch_size=batch, shuffle=False)

            testing_set = DNNData(x_test_torch, y_test_torch)
            testing_generator = DataLoader(testing_set, batch_size=len(testing_set), shuffle=False)

            valid_set = DNNData(x_valid_torch, y_valid_torch)
            valid_generator = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False)

            model = DNN(input_shape, all_tasks).to(device)
            path = "/home/jan/projects/EML/attack/1_attack/dump"

            ###### Pathways to save models 
            checkpoint_path = path + '/current_checkpoint.pt'

            #Path to saved model when train_epoch_loss <= train_loss_min
            bestmodel_path = path + '/best_model.pt'  

            #Path to saved model at minimum valid loss
            bestmodel_byvalid = path + '/best_model_by_valid.pt' 

            #Path to saved  when train_epoch_loss >= val_epoch_loss
            bestmodel_byvalid_crossed = path + '/best_model_by_valid-crossed.pt'   


            # ##### Train the neural network model

            criterion = torch.nn.BCELoss()

            # Optimizers require the parameters to optimize and a learning rate
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


            # Define the desired pathway
            writer = SummaryWriter('./dump/')


            ##################### With Tensorboard ######################
            loss_history=[]  
            correct_history=[]  
            val_loss_history=[]  
            val_correct_history=[] 
            train_loss_min = np.Inf
            val_loss_min = np.Inf


            # Training
            for e in range(train_epoch):
                
                model.train()
                # keep track of the loss over an epoch
                running_train_loss = 0
                running_valid_loss = 0
                running_train_correct = 0
                running_val_correct = 0
                y_train_true = []
                y_train_pred = []
                y_valid_true = []
                y_valid_pred = []
                batch = 0
                for x_batch, y_batch in training_generator:
                    batch += 1
                    if torch.cuda.is_available():
                        x_batch, y_batch = x_batch.cuda(), y_batch.cuda() 
                    
                    # Forward pass: Compute predicted y by passing x to the model
                    y_pred = model(x_batch)  # for all tasks
                    
                    # Compute loss over all tasks
                    loss = 0
                    correct = 0
                    y_train_true_task = []
                    y_train_pred_task = []
                    for i in range(len(all_tasks)):
                        y_batch_task = y_batch[:,i]
                        y_pred_task  = y_pred[i][:,0] #check if predictions na
                        # compute loss for labels that are not NA
                        indice_valid = y_batch_task >= 0
                        loss_task = criterion(y_pred_task[indice_valid], y_batch_task[indice_valid]) / N_train[i]
                        
                        loss += loss_task

                        pred_train = np.round(y_pred_task[indice_valid].detach().cpu().numpy())
                        target_train = y_batch_task[indice_valid].float()
                        y_train_true.extend(target_train.tolist()) 
                        y_train_pred.extend(pred_train.reshape(-1).tolist())

                    # Zero gradients, perform a backward pass, and update the weights.
                    writer.add_scalar("Accuracy/train", loss, batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    # sum up the losses from each batch
                    running_train_loss += loss.item()
                    writer.add_scalar("Loss/train", running_train_loss, e)
                    
                else:
                    with torch.no_grad():    
                    ## evaluation part 
                        model.eval()
                        for val_x_batch, val_y_batch in valid_generator:
                            
                            if torch.cuda.is_available():
                                val_x_batch, val_y_batch = val_x_batch.cuda(), val_y_batch.cuda() 
                            
                            val_output = model(val_x_batch)

                            ## 2. loss calculation over all tasks 
                            val_loss = 0
                            val_correct = 0
                            y_valid_true_task = []
                            y_valid_pred_task = []
                            for i in range(len(all_tasks)):
                                val_y_batch_task = val_y_batch[:,i]
                                val_output_task  = val_output[i][:,0]

                                # compute loss for labels that are not NA
                                indice_valid = val_y_batch_task >= 0
                                val_loss_task = criterion(val_output_task[indice_valid], val_y_batch_task[indice_valid]) / N_valid[i]

                                val_loss += val_loss_task
                                
                                pred_valid = np.round(val_output_task[indice_valid].detach().cpu().numpy())
                                target_valid = val_y_batch_task[indice_valid].float()
                                y_valid_true.extend(target_valid.tolist()) 
                                y_valid_pred.extend(pred_valid.reshape(-1).tolist())
                            

                            running_valid_loss+=val_loss.item()
                            writer.add_scalar("Loss/valid", running_valid_loss, e)
                    
                    #epoch loss
                    train_epoch_loss=np.mean(running_train_loss)
                    val_epoch_loss=np.mean(running_valid_loss)  
                
                    #epoch accuracy     
                    train_epoch_acc = accuracy_score(y_train_true,y_train_pred)
                    val_epoch_acc = accuracy_score(y_valid_true,y_valid_pred)
                    
                    #history
                    loss_history.append(train_epoch_loss)  
                    correct_history.append(train_epoch_acc)
                    val_loss_history.append(val_epoch_loss)  
                    val_correct_history.append(val_epoch_acc)  
                    
                    print("Epoch:", e, "Training Loss:", train_epoch_loss, "Valid Loss:", val_epoch_loss)
                    print("Training Acc:", train_epoch_acc, "Valid Acc:", val_epoch_acc)
                    
                    # create checkpoint variable and add important data
                    checkpoint = {
                        'epoch': e + 1,
                        'train_loss_min': train_epoch_loss,
                        'val_loss_min': val_epoch_loss, 
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    
                    # save checkpoint
                    save_ckp(checkpoint, False, checkpoint_path, bestmodel_path)
                    
                    ## TODO: save the model if validation loss has decreased
                    if train_epoch_loss <= train_loss_min:
                        print('Training loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,train_epoch_loss))
                        # save checkpoint as best model
                        save_ckp(checkpoint, True, checkpoint_path, bestmodel_path)
                        train_loss_min = train_epoch_loss
                        
                    if train_epoch_loss >= val_epoch_loss:
                        print('Training loss greater than validation loss ({:.6f} --> {:.6f}).  Saving model ...'.format(train_epoch_loss,val_epoch_loss))
                        # save checkpoint as best model
                        save_ckp(checkpoint, True, checkpoint_path, bestmodel_byvalid_crossed)
                        train_loss_min = train_epoch_loss
                        
                    if val_epoch_loss <= val_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(val_loss_min,val_epoch_loss))
                        # save checkpoint as best model
                        save_ckp(checkpoint, True, checkpoint_path, bestmodel_byvalid)
                        val_loss_min = val_epoch_loss



            # Loads model at lowest validation loss 
            loaded_model, optimizer, start_epoch, train_loss_min = load_ckp(bestmodel_byvalid, model, optimizer)


            # ##### Evaluate on test set
            # print test loss
            for x_test_torch, y_test_torch in testing_generator:
                y_test_pred = model.eval().to(device).cpu()(x_test_torch)
                
                # Compute loss over all tasks
                loss = 0
                for i in range(len(all_tasks)):
                    y_test_task = y_test_torch[:,i]
                    y_pred_task  = y_test_pred[i][:,0]

                    # compute loss for labels that are not NA
                    indice_valid = y_test_task >= 0
                    loss_task = criterion(y_pred_task[indice_valid], y_test_task[indice_valid]) / N_test[i]

                    loss += loss_task
                
            print(loss.item())


            results = {}
            # Collects performance metrics for all tasks on test set
            for i in range(len(all_tasks)):
                valid_datapoints = y_test[:,i] >= 0
                y_test_task = y_test[valid_datapoints,i] 
                y_test_pred_task = y_test_pred[i].detach().numpy()[valid_datapoints,0]
                
                acc = accuracy_score(y_test_task, np.round(y_test_pred_task))
                print('Accuracy for MTDNN on Morgan Fingerprint:', acc)
                
                bacc = sk.metrics.balanced_accuracy_score(y_test_task, np.round(y_test_pred_task))

                f1 = f1_score(y_test_task, np.round(y_test_pred_task), pos_label=1)
                print('F1 for MTDNN on Morgan Fingerprint:', f1)

                cfm = sk.metrics.confusion_matrix(y_test_task, np.round(y_test_pred_task))
                cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]

                tn, fp, fn, tp = cfm.ravel()
                pr = tp / (tp + fp)
                rc = tp / (tp + fn)
                print(' True Positive:', tp)
                print(' True Negative:', tn)
                print('False Positive:', fp)
                print('False Negative:', fn)
                
                
                auc = roc_auc_score(y_test_task, y_test_pred_task)
                print('Test ROC AUC ({}):'.format(all_tasks[i]), auc)
                

            print('Task'.ljust(10), '\t', '  AUC ', ' ACC ', ' BACC ', ' TN  ', ' TP  ', ' PR  ', ' RC  ', ' F1  ')
            for task, auc in results.items():
                print(task.ljust(10), '\t', np.round(auc,3))


            # ##### See Valid set performance
            # print test loss
            for x_valid_torch, y_valid_torch in valid_generator:
                y_valid_pred = model.eval().to(device).cpu()(x_valid_torch)
                
                # Compute loss over all tasks
                loss = 0
                for i in range(len(all_tasks)):
                    y_test_task = y_valid_torch[:,i]
                    y_pred_task  = y_valid_pred[i][:,0]

                    # compute loss for labels that are not NA
                    indice_valid = y_test_task >= 0
                    loss_task = criterion(y_pred_task[indice_valid], y_test_task[indice_valid]) / N_test[i]

                    loss += loss_task
                
            print(loss.item())



            results_valid = {}
            # Collects performance metrics for all tasks on Valid set
            for i in range(len(all_tasks)):
                
                valid_datapoints = y_valid[:,i] >= 0
                y_valid_task = y_valid[valid_datapoints,i] 
                y_valid_pred_task = y_valid_pred[i].detach().numpy()[valid_datapoints,0]
                
                
                acc = accuracy_score(y_valid_task, np.round(y_valid_pred_task))
                print('Accuracy for deepnn on Morgan Fingerprint:', acc)
                
                bacc = sk.metrics.balanced_accuracy_score(y_valid_task, np.round(y_valid_pred_task))

                f1 = f1_score(y_valid_task, np.round(y_valid_pred_task), pos_label=1)
                print('F1 for deepnn on Morgan Fingerprint:', f1)

                cfm = sk.metrics.confusion_matrix(y_valid_task, np.round(y_valid_pred_task))
                cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]

                print('Confusion Matrix for deepnn on Morgan Fingerprint:\n', cfm)

                tn, fp, fn, tp = cfm.ravel()
                pr = tp / (tp + fp)
                rc = tp / (tp + fn)
                print(' True Positive:', tp)
                print(' True Negative:', tn)
                print('False Positive:', fp)
                print('False Negative:', fn)
                
                
                auc = roc_auc_score(y_valid_task, y_valid_pred_task)
                print('Test ROC AUC ({}):'.format(all_tasks[i]), auc)
                
                results_valid[all_tasks[i]] = [auc, acc, bacc, tn, tp, pr, rc, f1]

                fpr, tpr, threshold = sk.metrics.roc_curve(y_valid_task, y_valid_pred_task)
                plt.plot(fpr, tpr, 'b', label = 'AUC')
                plt.legend(loc = 'lower right')
                plt.plot([0, 1], [0, 1],'r--')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.show()



            print('Task'.ljust(35), '\t', '  AUC ', ' ACC ', ' BACC ', ' TN  ', ' TP  ', ' PR  ', ' RC  ', ' F1  ')
            for task, auc in results_valid.items():
                print(task.ljust(35), '\t', np.round(auc,3))