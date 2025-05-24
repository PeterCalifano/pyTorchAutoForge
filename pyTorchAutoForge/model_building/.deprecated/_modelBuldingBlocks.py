from pyTorchAutoForge.api.torch import *
from pyTorchAutoForge.model_building.ModelAutoBuilder import AutoComputeConvBlocksOutput, ComputeConv2dOutputSize, ComputePooling2dOutputSize, ComputeConvBlockOutputSize, EnumMultiHeadOutMode, MultiHeadRegressor
from typing import Literal

from pyTorchAutoForge.utils import GetDeviceMulti
from pyTorchAutoForge.setup import BaseConfigClass
from pyTorchAutoForge.model_building.modelBuildingFunctions import build_activation_layer
from pyTorchAutoForge.model_building.ModelMutator import ModelMutator
from pyTorchAutoForge.model_building.convolutionalBlocks import ConvolutionalBlock1d, ConvolutionalBlock2d, ConvolutionalBlock3d

from torch import nn
from torch.nn import functional as torchFunc
import torch
import optuna
import os
import kornia

from dataclasses import dataclass
import numpy as np
from torchvision import models
from abc import ABC
from typing import Literal

# %% TEMPORARY DEV
# TODO: make this function generic!
def ReloadModelFromOptuna(trial: optuna.trial.FrozenTrial, other_params: dict, modelName: str, filepath: str) -> nn.Module:

    num_of_epochs = 125
    # other_params = dict()

    # Sample decision parameters space
    # Optimization strategy
    initial_lr = trial.suggest_float('initial_lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 2, 40, step=8)

    # initial_lr = 5e-4
    # other_params['initial_lr'] = initial_lr
    # batch_size = 20
    # other_params['batch_size'] = batch_size

    # Model
    # use_default_size = trial.suggest_int('use_default_size', 0, 1)
    # efficient_net_ID = trial.suggest_int('efficient_net_ID', 0, 1)

    try:
        regressor_arch_version = trial.suggest_int(
            'regressor_arch_version', 1, 2)
    except:
        other_params['regressor_arch_version'] = 1
        regressor_arch_version = other_params['regressor_arch_version']

    # dropout_coeff_multiplier = trial.suggest_int('dropout_coeff_multiplier', 0, 10, step=1)
    dropout_coeff_multiplier = 0
    use_batchnorm = 0

    try:
        image_adapter_strategy = trial.suggest_categorical(
            'image_adapter_strategy', ['resize_copy', 'conv_adapter'])
    except:
        other_params['image_adapter_strategy'] = 'resize_copy'
        image_adapter_strategy = other_params['image_adapter_strategy']

    try:
        mutate_to_groupnorm = trial.suggest_int('mutate_to_groupnorm', 0, 1)
    except:
        other_params['mutate_to_groupnorm'] = 1
        mutate_to_groupnorm = other_params['mutate_to_groupnorm']

    loss_type = trial.suggest_categorical('loss_type', ['mse', 'huber'])

    # use_batchnorm = trial.suggest_int('use_batchnorm', 0, 1)
    num_of_regressor_layers_H1 = trial.suggest_int(
        'num_of_regressor_layers_H1', 3, 7)
    num_of_regressor_layers_H2 = trial.suggest_int(
        'num_of_regressor_layers_H2', 3, 7)

    scheduler = trial.suggest_categorical(
        'lr_scheduler_name', ['cosine_annealing_restarts', 'exponential_decay'])

    if scheduler == 'cosine_annealing_restarts':
        T0_WarmAnnealer = trial.suggest_int('T0_WarmAnnealer', np.floor(
            0.85 * num_of_epochs), 3*num_of_epochs, step=5)
        lr_min = trial.suggest_float('lr_min', 1e-8, 1e-5, log=True)

    elif scheduler == 'exponential_decay':
        gamma = trial.suggest_float('gamma', 0.900, 0.999, step=0.005)

    # Define regressor architecture for centroid prediction
    out_channels_sizes_H1 = []
    out_channels_sizes_H1.append(2)  # Output layer

    for i in range(num_of_regressor_layers_H1):
        out_channels_sizes_H1.append(2**(i+5))

    out_channels_sizes_H1.reverse()
    other_params['out_channels_sizes_H1'] = out_channels_sizes_H1

    out_channels_sizes_H2 = []
    out_channels_sizes_H2.append(2)  # Output layer

    # Define regressor architecture for range prediction
    for i in range(num_of_regressor_layers_H2):
        out_channels_sizes_H2.append(2**(i+5))

    out_channels_sizes_H2.reverse()
    other_params['out_channels_sizes_H2'] = out_channels_sizes_H2

    # Build regression layers with decreasing number of channels as powers of 2
    other_params['model_definition_mode'] = 'multihead'
    model_definition_mode = other_params['model_definition_mode']

    # Print the parameters
    print(f"Parameters: \n"
          f"initial_lr: {initial_lr}\n"
          f"batch_size: {batch_size}\n"
          f"dropout_coeff_multiplier: {dropout_coeff_multiplier}\n"
          f"use_batchnorm: {use_batchnorm}\n"
          f"mutate_to_groupnorm: {mutate_to_groupnorm}\n"
          f"loss_type: {loss_type}\n"
          f"num_of_regressor_layers_H1: {num_of_regressor_layers_H1}\n"
          f"num_of_regressor_layers_H2: {num_of_regressor_layers_H2}\n"
          f"lr_scheduler_name: {scheduler}\n"
          f"out_channels_sizes_H1: {out_channels_sizes_H1}\n"
          f"out_channels_sizes_H2: {out_channels_sizes_H2}\n"
          f"model_definition_mode: {model_definition_mode}\n"
          f"image_adapter_strategy: {image_adapter_strategy}\n"
          f"regressor_arch_version: {regressor_arch_version}\n")

    if scheduler == 'cosine_annealing_restarts':
        print(f"T0_WarmAnnealer: {T0_WarmAnnealer}\n"
              f"lr_min: {lr_min}\n")
    elif scheduler == 'exponential_decay':
        print(f"gamma: {gamma}\n")

    # Define model
    model = DefineModel(trial, other_params)

    # Load model parameters
    model = LoadModel(model, os.path.join(filepath, modelName), False)

    # Loading validation
    ValidateDictLoading(model, modelName, filepath)

    return model
