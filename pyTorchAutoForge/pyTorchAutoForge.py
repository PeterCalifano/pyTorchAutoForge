'''
Module collecting utilities function building upon PyTorch to speed up prototyping, training and testing of Neural Nets.
Utilities for monitoring training, validation and testing of models, as well as saving and loading models and datasets are included.
The module also includes functions to export and load models to/from ONNx format, as well as a MATLAB wrapper class for model evaluation.
NOTE: This however, requires TorchTCP module (not included yet) and a MATLAB tcpclient based interface.
Created by PeterC - 04-05-2024. Current version: v0.1 (30-06-2024)
'''

# Import modules
from typing import Optional, Any, Union
import torch, mlflow, optuna
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)

# import datetime
import numpy as np
import os, copy
from typing import Union



