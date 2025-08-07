"""
_summary_

_extended_summary_
"""
from re import S
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Import classes from PTAF
from pyTorchAutoForge.datasets import DataloaderIndex
from pyTorchAutoForge.optimization import ModelTrainingManagerConfig, ModelTrainingManager, TaskType
from pyTorchAutoForge.model_building import ModelMutator
from pyTorchAutoForge.utils import GetDevice

# First, let's define a super simple NN model.
# In PTAF, any model is fine as long as it is torch.nn.Module "compatible". 
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define some dummy data
dummy_X = torch.randn(1000, 20)  # 1000 samples, 20 features
dummy_y = torch.randn(1000, 2)   # 1000 samples, 2 targets


# %% Define the dataset to build the index
# For simple tasks like vector to vector regression, PTAF provides a "RegressionDataset" class. 
batch_size = 512

train_dataset = TensorDataset(dummy_X, dummy_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Then, define the dataloader index. This class allows one to store training and validation dataloader in a single object, whereas a random split can be used in case only the first is provided. 
dataloader_index = DataloaderIndex(train_loader, split_ratio=0.9)

# Define a loss function as the standard torch ones. 
# Any function can be used, as long as it is torch-like, one tensor as input and one tensor as output. 
lossFcn = torch.nn.HuberLoss(delta=0.001)  # Changed to Huber Loss

# Setup the trainer configuration
# This can be done either by manually defining the configuration object, by loading it from the corresponding yaml file, or lastly by using the default with modifiers from input arguments, through the parser (WIP).
initial_lr = 1e-2
step_size_LR = 10
num_of_epochs = 100

# Define the model
model = SimpleNN(input_size=20, output_size=2)  # Example input size and output size

# The optimizers are essentially the same as in torch, PTAF currently only provides boilerplate code to automate their usage and logging.
optimizer = torch.optim.AdamW(model.parameters(), amsgrad=False, lr=initial_lr, weight_decay=1E-8, fused=True)

# You may also want to specify a scheduler for the learning rate. None is used by default.
lr_scheduler_manual = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_of_epochs, eta_min=5e-7, last_epoch=-1)

# Lastly, the trainer configuration object is created.
trainerConfig = ModelTrainingManagerConfig(
    batch_size=batch_size,
    tasktype=TaskType.REGRESSION, # Two possible values: REGRESSION or CLASSIFICATION 
    initial_lr=initial_lr,
    lr_scheduler=lr_scheduler_manual,
    num_of_epochs=num_of_epochs,
    optimizer=optimizer,
    eval_example=True, # Set whether you want intermediate evaluation of the model on the validation datasets, with some statistics
    checkpoint_dir="checkpoints", # Specify the directory where the checkpoints will be saved. Created if not existing.
    optuna_trial=None, # This is a special entry for hyperparameter optimization with optuna, do not consider for now
    device=GetDevice(),
    mlflow_logging=True) # Note: currently, checkpoints loading is not yet automated, but soon it will ;)

# Set the trainer to get it ready for training and validation
# You can also change some parameters directly as input arguments instead of the configuration object
# Anyhow, a summary will be printed prior to starting the training so you can check if everything is correct.
trainer = ModelTrainingManager( model=model, 
                               lossFcn=lossFcn, 
                               config=trainerConfig,
                                paramsToLogDict={"model": model} 
                                )

print("\nModelTrainingManager instance:", trainer)

# Note here that by design, mlflow is automatically used for logging, locally if not set otherwise. You do not really need to do anything. Just open the mlflow UI and see the training process. In shell, go to the parent directory where mlflow_runs folder gets created and type "mlflow ui". A server will start on localhost:5000. You may want to log something more that the default values, just to make sure to distinguish between different runs. Well, it is simple as adding entries in a dictionary passed with the "paramsToLogDict" kwarg.

# Last but not least, set the dataloader index and start the training.
trainer.setDataloaders(dataloader_index)
# By default, the best model found during the training is returned. This is regulated by the flag "keep_best" in the configuration.
# The checkpoints saved are always at the current epoch however, while the trainer can save a traced model or a state dict of the best once finished, which is then returned here
model = trainer.trainAndValidate()

