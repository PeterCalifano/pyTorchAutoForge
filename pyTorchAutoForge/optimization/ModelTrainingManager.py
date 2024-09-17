

# TODO Add yaml interface for training, compatible with mlflow and optuna
# The idea is to let the user specify all the parameters in a yaml file, which is then loaded and used
# to set the configuration class. Default values are specified as the class defaults. 
# Loading methods only modify the parameters the user has specified


from typing import Optional, Any, Union
import torch
import mlflow
import optuna
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
# Utils for dataset management, storing pairs of (sample, label)
from torch.utils.data import DataLoader
from torchvision import datasets  # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor  # Utils
from dataclasses import dataclass, asdict, fields, Field, MISSING

# import datetime
import numpy as np
import sys
import os
import signal
import copy
import inspect
import subprocess
import psutil
import onnx
import yaml
from onnx import version_converter
from typing import Union, IO

# Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
# Module to apply activation functions in forward pass instead of defining them in the model class
import torch.nn.functional as F


# %% Training and validation manager class - 22-06-2024 (WIP)
# TODO: Features to include:
# 1) Multi-process/multi-threading support for training and validation of multiple models in parallel
# 2) Logging of all relevat options and results to file (either csv or text from std output)
# 3) Main training logbook to store all data to be used for model selection and hyperparameter tuning, this should be "per project"
# 4) Training mode: k-fold cross validation leveraging scikit-learn

@dataclass(frozen=True)
class ModelTrainingManagerConfig():
    '''Configuration dataclass for ModelTrainingManager class. Contains all parameters ModelTrainingManager accepts as configuration.'''
    
    # DATA fields with default values
    initial_lr: float = 1e-4
    lr_scheduler: Any = None
    optim_momentum: float = 0.5  # Momentum value for SGD optimizer
    
    # Define default optimizer as Adam
    optimizer: Any = torch.optim.Adam(lr=initial_lr)

    # DEVNOTE: dataclass generates __init__() automatically
    # Same goes for __repr()__ for printing and __eq()__ for equality check methods

    def getConfigDict(self) -> dict:
        '''Method to return the dataclass as dictionary'''
        return asdict(self)

    #def display(self) -> None:
    #    print('ModelTrainingManager configuration parameters:\n\t', self.getConfig())

    @classmethod
    # DEVNOTE: classmethod is like static methods, but apply to the class itself and passes it implicitly as the first argument
    def load_from_yaml(cls, yamlFile: Union[str | IO]) -> 'ModelTrainingManagerConfig':
        '''Method to load configuration parameters from a yaml file containing configuration dictionary'''

        if isinstance(yamlFile, str):
            with open(yamlFile, 'r') as file:

                # TODO: VALIDATE SCHEMA

                # Parse yaml file to dictionary
                configDict = yaml.safe_load(file)
        else:

            # TODO: VALIDATE SCHEMA

            configDict = yaml.safe_load(yamlFile)

        # Call load_from_dict() method
        return cls.load_from_dict(configDict)

    @classmethod
    def load_from_dict(cls, configDict: dict) -> 'ModelTrainingManagerConfig':
        '''Method to load configuration parameters from a dictionary'''

        # Get all field names from the class
        fieldNames = {f.name for f in fields(cls)}
        # Get fields in configuration dictionary
        missingFields = fieldNames - configDict.keys()

        # Check if any required field is missing (those without default values)
        requiredFields = {f.name for f in fields(
            cls) if f.default is MISSING and f.default_factory is MISSING}
        missingRequired = requiredFields & missingFields

        if missingRequired:
            raise ValueError(f"Config dict is missing required fields: {missingRequired}")
        
        # Build initialization arguments for class (using autogen __init__() method)
        # All fields not specified by configDict are initialized as default from cls values
        initArgs = {key: configDict.get( key, getattr(cls, key)) for key in fieldNames}
        
        # Return instance of class with attributes defined from dictionary
        return cls(**initArgs)
    
    @staticmethod
    def getConfigParamsNames(cls) -> list:
        '''Method to return the names of all parameters in the configuration class'''
        return [f.name for f in fields(cls)]
    
        

# %% Function to get the number of trainable parameters in a model - 11-06-2024


def getNumOfTrainParams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %% ModelTrainingManager class - 24-07-2024
# DEV NOTE: inheritance from config?


class ModelTrainingManager(ModelTrainingManagerConfig):
    '''Class to manage training and validation of PyTorch models using specified datasets and loss functions.'''

    def __init__(self, model: nn.Module, lossFcn: nn.Module, 
                 optimizer:Union[optim.Optimizer, int], options: Union[ModelTrainingManagerConfig, dict]) -> None:
        '''Constructor for TrainAndValidationManager class. Initializes model, loss function, optimizer and training/validation options.'''
        super().__init__(options.initial_lr, options.lr_scheduler)
        
        # Define manager parameters
        self.model = model
        self.lossFcn = lossFcn

        # Optimizer --> # TODO: check how to modify learning rate and momentum while training
        if isinstance(optimizer, optim.Optimizer):
            self.optimizer = optimizer

        elif isinstance(optimizer, int):
            if optimizer == 0:
                optimizer = torch.optim.SGD(
                     self.model.parameters(), lr=learnRate, momentum=momentumValue)
            elif optimizer == 1:
                optimizer = torch.optim.Adam(
                     self.model.parameters(), lr=learnRate)
            else:
                raise ValueError(
                     'Optimizer type not recognized. Use either 0 for SGD or 1 for Adam.')
        else:
            raise ValueError(
                'Optimizer must be either an instance of torch.optim.Optimizer or an integer representing the optimizer type.')

        # Define training and validation options

    def LoadDatasets(self, dataloaderIndex: dict):
        '''Method to load datasets from dataloaderIndex and use them depending on the specified criterion (e.g. "order", "merge)'''
        # TODO: Load all datasets from dataloaderIndex and use them depending on the specified criterion (e.g. "order", "merge)
        pass

    def GetTracedModel(self):
        pass


