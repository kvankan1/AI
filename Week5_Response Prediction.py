#%% imports
import os
import random
import sys
import time
from typing import Callable, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from exe_4_utils import filter_plot_losses, predict, setup_predictions_plot, split_scale, weights_init
from read_data import get_composite_file_names, generate_data_from_files

from library import EarlyStopper



class DatasetExe4(Dataset):
    """Implement this dataset as a standard Pytorch dataset.
    Make sure to include all the necessary methods.
    X and y are the PyTorch tensors containing your data.
    
    Ref: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files"""
    def __init__(self, X, y) -> None:
        super().__init__()
        self._X = X
        self._y = y
    
    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        return self._X[idx], self._y[idx]
        

def train_model_early_stop(model:nn.Module, train_loader:DataLoader, X_val:torch.tensor, y_val:torch.tensor, loss_function: Callable, optimizer: torch.optim.Optimizer, n_epochs: int = 500, tol_train: float = 1e-5, es_patience=1, es_delta=0., verbose: bool = False):
    """Train the model with early stopping, while updating the lists 'train_loss_history' and 'val_loss_history'
    with the training and validation loss at each epoch.
    The model is trained on batches, which are iterated through a dataloader ('train_loader' in the input).
    The function should return training and validation losses during the epochs and the mean wall-clock time elapsed per epoch."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_loss_history = []
    val_loss_history = []
    es = EarlyStopper(patience=es_patience, min_delta=es_delta)
    n_batches = len(train_loader)
    start_time = time.time()
    
    for epoch in range(n_epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_batches = 0
        
        for X_train, y_train in train_loader:
            # move data to device
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred_train = model(X_train)
            loss_train = loss_function(y_pred_train, y_train)
            loss_train.backward()
            optimizer.step()

            # add the loss to the epoch's training loss
            train_loss += loss_train.item()
            train_batches += 1
        
        # calculate the average training loss for the epoch
        train_loss /= train_batches
        train_loss_history.append(train_loss)
        
        # calculate the validation loss for the epoch
        with torch.no_grad():
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            y_pred_val = model(X_val)
            loss_val = loss_function(y_pred_val, y_val)
            val_loss_history.append(loss_val.item())

            # check early stopping criterion
            if es.early_stop(loss_val.item()):
                print(f"Early stopping at epoch {epoch}")
                break
        
        # print epoch info if verbose flag is set
        if verbose:
            print(f"Epoch {epoch+1:2d} - Training loss: {train_loss:.6f}, Validation loss: {loss_val:.6f}")
        
    end_time = time.time()
    mean_epoch_time = (end_time - start_time) / (epoch + 1)
    
    return train_loss_history, val_loss_history, mean_epoch_time








