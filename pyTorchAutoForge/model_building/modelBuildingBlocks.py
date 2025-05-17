# Module to apply activation functions in forward pass instead of defining them in the model class
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
import torch, optuna, os, kornia

from dataclasses import dataclass   
import numpy as np
from torchvision import models
from abc import ABC
from typing import Literal

# DEVNOTE TODO change name of this file to "modelBuildingBlocks.py" and move the OLD classes to the file "modelClasses.py" for compatibility with legacy codebase
 
#############################################################################################################################################
class AutoForgeModule(torch.nn.Module):
    """
    AutoForgeModule Custom base class inheriting nn.Module to define a PyTorch NN model, augmented with saving/loading routines like Pytorch Lightning.

    _extended_summary_

    :param torch: _description_
    :type torch: _type_
    :raises Warning: _description_
    """
    
    def __init__(self, moduleName : str | None = None, enable_tracing : bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Assign module name. If not provided by user, use class name
        if moduleName is None:
            self.moduleName = self.__class__.__name__
        else:
            self.moduleName  = moduleName


    def save(self, exampleInput = None, target_device : str | None = None) -> None:

        if self.enable_tracing == True and exampleInput is None:
            self.enable_tracing = False
            raise Warning('You must provide an example input to trace the model through torch.jit.trace(). Overriding enable_tracing to False.')
        
        if target_device is None:
            target_device = self.device

    def load(self):
        pass
        #LoadModel()
#############################################################################################################################################
# TBC: class to perform code generation of net classes instead of classes with for and if loops? 
# --> THe key problem with the latter is that tracing/scripting is likely to fail due to conditional statements

# TODO The structure of the model building blocks should be as follows:
# Normalization Layer example:

###########################################################
@dataclass
class TemplateNetBaseConfig(BaseConfigClass):

    # General
    model_name: str = "template_network"

    # Architecture design
    regularization_layer_type: Literal['batchnorm',
                                       'dropout', 'groupnorm'] = 'batchnorm'
    
    out_channels_sizes : list[int] | None = None

# %% TemplateConvNet2d - 19-09-2024
@dataclass 
class TemplateConvNetConfig(TemplateNetBaseConfig):

    # Generic convolutional blocks parameters
    # Pooling parameters
    pool_type: Literal[
        "MaxPool1d", "AvgPool1d", "Adapt_MaxPool1d", "Adapt_AvgPool1d",
        "MaxPool2d", "AvgPool2d", "Adapt_MaxPool2d", "Adapt_AvgPool2d",
        "MaxPool3d", "AvgPool3d", "Adapt_MaxPool3d", "Adapt_AvgPool3d",
        "none"
    ] = "MaxPool2d"
    
    activ_type: Literal["prelu", "sigmoid",
                        "relu", "tanh", "none"] = "prelu"
    
    regularizer_type: Literal["dropout",
                              "batchnorm", "groupnorm", "none"] = "none"
    regularized_param : int | float = 0.0
    
    conv_stride: int | tuple[int, int, int] = 1
    conv_padding: int | tuple[int, int, int] = 0
    conv_dilation: int | tuple[int, int, int] = 1
    prelu_params: Literal["all", "unique"] = "unique"
    
    # Nominal size of input tensor. Optional to verify design can work
    reference_input_size: tuple[int, ...] | None = None

@dataclass
class TemplateConvNetConfig2d(TemplateConvNetConfig):

    save_intermediate_features : bool = False

    kernel_sizes: list[int] | None = None
    pool_kernel_sizes: list[int] | int | None = None
    num_input_channels: int = 3  # Default is 3

    output_linear_layer_size : int | None = None # If specified, a convolution using out_channels as this number is added # TODO

    add_fcn_layer_size : int | None = None # By default, no linear output layer
    size_skip_to_linear_output: int | None = None


    def __post_init__(self):

        # Check pooling type is correct (2d)
        if not self.pool_type.endswith("2d"):
            raise TypeError(f"TemplateConvNetConfig2d: pool_type must be of type 'MaxPool2d', 'AvgPool2d', 'Adapt_MaxPool2d' or 'Adapt_AvgPool2d'. Found {self.pool_type}.")

        # Check config validity, throw error is not
        if self.kernel_sizes is None:
            raise ValueError("TemplateConvNetConfig2d: 'kernel_sizes' cannot be None")
        if self.pool_kernel_sizes is None:
            raise ValueError("TemplateConvNetConfig2d: 'pool_kernel_sizes' cannot be None")
        if self.out_channels_sizes is None:
            raise ValueError("TemplateConvNetConfig2d: 'out_channels_sizes' cannot be None")
            
        if len(self.kernel_sizes) != len(self.out_channels_sizes):
            raise ValueError("TemplateConvNetConfig2d: 'kernel_sizes' and 'out_channels_sizes' must have the same length")

        if isinstance(self.pool_kernel_sizes, list):
            if len(self.kernel_sizes) != len(self.pool_kernel_sizes):
                raise ValueError("TemplateConvNetConfig2d: 'kernel_sizes' and 'pool_kernel_sizes' must have the same length. Alternatively, pool_kernel_sizes must be scalar integer.")
            
        # Automagic configuration post-processing
        # If pooling kernel size is scalar, unroll to number of layers
        if isinstance(self.pool_kernel_sizes, int):
            self.pool_kernel_sizes = [
                self.pool_kernel_sizes] * len(self.kernel_sizes)

        assert( isinstance(conv_stride, int) ), "conv_stride must be a scalar integer for ConvolutionalBlock2d"

        # TODO add check on conv sizes if reference_input_size is passed, to ensure kernel and pool sizes are compatible
        # convBlockOutputSize = AutoComputeConvBlocksOutput( self, kernel_sizes, pool_kernel_sizes)

class TemplateConvNet2d(AutoForgeModule):
    '''
    Template class for a fully parametric CNN model in PyTorch. Inherits from AutoForgeModule class (nn.Module enhanced class).
    '''
    
    def __init__(self, cfg: TemplateConvNetConfig2d) -> None:
        super().__init__()

        self.cfg = cfg

        # Build architecture model
        kernel_sizes = cfg.kernel_sizes
        pool_kernel_sizes = cfg.pool_kernel_sizes

        if kernel_sizes is None or pool_kernel_sizes is None:
            raise ValueError(
                'Kernel and pooling kernel sizes must not be none')

        if isinstance(pool_kernel_sizes, list):
            if len(kernel_sizes) != len(pool_kernel_sizes):
                raise ValueError(
                    'Kernel and pooling kernel sizes must have the same length')
        else:
            raise ValueError('pool_kernel_sizes cannot be scalar')

        # Define output layer if required by config
        self.linear_output_layer : nn.Module | None

        # Model parameters
        self.out_channels_sizes = cfg.out_channels_sizes
        self.num_of_conv_blocks = len(kernel_sizes)

        # Additional checks
        if self.out_channels_sizes is None:
            raise ValueError(
                'TemplateConvNetConfig2d: out_channels_sizes cannot be None')

        self.blocks = nn.ModuleList()

        # Model architecture
        idLayer = 0

        # Convolutional blocks auto building
        in_channels = cfg.num_input_channels    
        for ith in range(len(kernel_sizes)):

            # Get data for ith block
            kernel_size = kernel_sizes[ith]
            pool_kernel_size = pool_kernel_sizes[ith]
            out_channels = self.out_channels_sizes[ith]
            pool_type = cfg.pool_type
            activ_type_ = cfg.activ_type
            regularization_layer_type_ = cfg.regularization_layer_type
            regularized_param_ = cfg.regularized_param
            conv_stride_ = cfg.conv_stride

            # Convolutional blocks
            block = ConvolutionalBlock2d(in_channels=in_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=kernel_size,
                                        pool_kernel_size=pool_kernel_size,
                                        pool_type=pool_type, #type:ignore
                                        activ_type=activ_type_,
                                        regularizer_type=regularization_layer_type_,
                                        regularized_param=regularized_param_,
                                        conv_stride=conv_stride_, #type:ignore
                                        )

            self.blocks.append(block)

            in_channels = out_channels
            idLayer += 1

        if cfg.add_fcn_layer_size is not None:
            
            regressor_sequential = nn.ModuleList()
            if in_channels != cfg.add_fcn_layer_size:
                # Add convolutional "expander"
                regressor_sequential.append(nn.Conv2d(in_channels, cfg.add_fcn_layer_size, 1, 1))

            # Fully Connected regressor
            regressor_sequential.append(nn.AdaptiveAvgPool2d((1,1))) 
            regressor_sequential.append(module=nn.Flatten())
            regressor_sequential.append(nn.Linear(in_channels, 
            cfg.add_fcn_layer_size, bias=True))

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''

        for layer in self.blocks:
            # Check if layer is a Linear layer
            if isinstance(layer, nn.Linear):
                # Apply Kaiming initialization
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    # Initialize bias to zero if present
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.Conv2d):
                # Apply Kaiming initialization
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    # Initialize bias to zero if present
                    nn.init.constant_(layer.bias, 0)

    def forward(self, X, X_skips: list[torch.Tensor] | torch.Tensor | None = None):
        """
        Generic forward pass for TemplateConvNet2d. Use only as prototype. Graph capture, scripting and tracing will likely not work with this class.
        """

        # Perform forward pass iterating through all blocks of CNN
        x = X

        # TODO upgrade template to receive a second input: feature map skip
        # TODO add how to merge x_skip with x

        x_skip_out = []
        if isinstance(X_skips, (torch.Tensor, type(None))):

            for block in self.blocks:
                x = block(x)
                if self.cfg.save_intermediate_features:
                    x_skip_out.append(x)

        elif isinstance(X_skips, list):

            for block, x_skip in zip(self.blocks, X_skips):
                x = block(x)
                if self.cfg.save_intermediate_features:
                    x_skip_out.append(x)
        
        return x, x_skip_out


# %% TemplateFullyConnectedDeepNetConfig - 19-09-2024
@dataclass
class TemplateFullyConnectedDeepNetConfig(TemplateNetBaseConfig):

    # Architecture definition
    out_channel_sizes: list[int] | None = None
    input_layer_size: int | None = None
    dropout_probability : float = 0.0

    def __post_init__(self):
        def __post_init__(self):
            if self.out_channel_sizes is None:
                raise ValueError("TemplateFullyConnectedDeepNetConfig: 'out_channel_sizes' cannot be None")
            if self.input_layer_size is None:
                raise ValueError("TemplateFullyConnectedDeepNetConfig: 'input_layer_size' cannot be None")

class TemplateFullyConnectedDeepNet(AutoForgeModule):
    '''
    Template class for a fully parametric Deep NN model in PyTorch. Inherits from AutoForgeModule class (nn.Module enhanced class).
    '''

    def __init__(self, cfg : TemplateFullyConnectedDeepNetConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.model_name = cfg.model_name

        regularization_layer_type = cfg.regularization_layer_type
        dropout_probability = cfg.dropout_probability

        if regularization_layer_type == 'batchnorm':
            self.use_batchnorm = True

        elif regularization_layer_type == 'dropout':
            self.use_batchnorm = False

        elif regularization_layer_type == 'groupnorm':
            raise NotImplementedError(
                'Group normalization is not implemented yet. Please use batch normalization or dropout instead.')
        else:
            self.dropout_probability = 0.0
            self.use_batchnorm = False

        out_channel_sizes = cfg.out_channel_sizes

        # Initialize input size for first layer
        input_size = cfg.input_layer_size

        # Model parameters
        self.out_channel_sizes = out_channel_sizes
        self.use_batchnorm = self.use_batchnorm

        self.num_layers = len(self.out_channel_sizes)

        # Model architecture
        self.layers = nn.ModuleList()
        idLayer = 0

        # Fully Connected autobuilder
        self.layers.append(nn.Flatten())

        for i in range(idLayer, self.num_layers+idLayer-1):

            # Fully Connected layers block
            self.layers.append(nn.Linear(input_size, self.out_channel_sizes[i], bias=True))
            self.layers.append(nn.PReLU(self.out_channel_sizes[i]))

            # Dropout is inhibited if batch normalization
            if not self.use_batchnorm and dropout_probability > 0:
                self.layers.append(nn.Dropout(dropout_probability))

            # Add batch normalization layer if required
            if self.use_batchnorm:
                self.layers.append(nn.BatchNorm1d(
                    self.out_channel_sizes[i], eps=1E-5, momentum=0.1, affine=True))

            # Update input size for next layer
            input_size = self.out_channel_sizes[i]

        # Add output layer
        self.layers.append(nn.Linear(input_size, self.out_channel_sizes[-1], bias=True))

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''

        for layer in self.layers:
            # Check if layer is a Linear layer
            if isinstance(layer, nn.Linear):
                # Apply Kaiming initialization
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    # Initialize bias to zero if present
                    nn.init.constant_(layer.bias, 0)

    def forward(self, inputSample):
        # Perform forward pass iterating through all layers of DNN
        val = inputSample
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                val = layer(val)
            elif isinstance(layer, nn.PReLU):
                val = torchFunc.prelu(val, layer.weight)
            elif isinstance(layer, nn.Dropout):
                val = layer(val)
            elif isinstance(layer, nn.BatchNorm1d):
                val = layer(val)
            elif isinstance(layer, nn.Flatten):
                val = layer(val)

        # Output layer
        prediction = val

        return prediction


# DEVELOPMENT CODE: DEVNOTE: test definition of template DNN using new build_activation_layer function
class TemplateFullyConnectedDeepNetConfig_experimental(AutoForgeModule):
    '''Template class for a fully parametric Deep NN model in PyTorch. Inherits from AutoForgeModule class (nn.Module enhanced class).'''

    def __init__(self, parametersConfig) -> None:
        super().__init__()

        useBatchNorm = parametersConfig.get('useBatchNorm', False) # TODO try to replace with build_normalization_layer function
        alphaDropCoeffLayers = parametersConfig.get('alphaDropCoeffLayers', None) # Can be either scalar (apply to all) or list (apply to specific layers)
        #alphaLeaky = parametersConfig.get('alphaLeaky', 0)
        out_channels_sizes = parametersConfig.get('out_channels_sizes', [])

        if alphaDropCoeffLayers is not None:
            assert len(alphaDropCoeffLayers) == len(out_channels_sizes) -1, 'Length of alphaDropCoeffLayers must match number of layers in out_channels_sizes'

        # Define activation function parameters (default: PReLU)
        self.activation_fcn_name = parametersConfig.get( 'activation_fcn_name', 'PReLU')
        act_fcn_params_dict = parametersConfig.get( 'act_fcn_params_dict', {'num_parameters': 'all'})

        # Initialize input size for first layer
        input_size = parametersConfig.get('input_size')

        # Model parameters
        self.out_channels_sizes = out_channels_sizes
        self.useBatchNorm = useBatchNorm

        self.num_layers = len(self.out_channels_sizes)

        # Model architecture
        self.layers = nn.ModuleList()
        idLayer = 0

        # Fully Connected autobuilder
        self.layers.append(nn.Flatten())

        for i in range(idLayer, self.num_layers+idLayer-1):

            # Build Linear layer
            self.layers.append( nn.Linear(input_size, self.out_channels_sizes[i], bias=True))

            # Build activation layer
            if self.activation_fcn_name == 'PReLU': 
                act_fcn_params_dict['num_parameters'] = self.out_channels_sizes[i]

            self.layers.append(build_activation_layer( self.activation_fcn_name, False, **act_fcn_params_dict))

            # Add dropout layer if required
            if alphaDropCoeffLayers is not None:
                if len(alphaDropCoeffLayers) > 0 and len(alphaDropCoeffLayers) == 1:
                    self.layers.append(nn.Dropout(alphaDropCoeffLayers[0])) # Add to all layers

                if alphaDropCoeffLayers[i] > 0:
                    self.layers.append(nn.Dropout(alphaDropCoeffLayers[i])) # Add to layer as specified by user

            # Add batch normalization layer if required
            if self.useBatchNorm:
                self.layers.append(nn.BatchNorm1d( self.out_channels_sizes[i], eps=1E-5, momentum=0.1, affine=True))

            # Update input size for next layer
            input_size = self.out_channels_sizes[i]

        # Add output layer
        self.layers.append(nn.Linear(input_size, self.out_channels_sizes[-1], bias=True))

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''

        for layer in self.layers:

            # Check if layer is a Linear layer
            if isinstance(layer, nn.Linear):
                # Apply Kaiming initialization
                if self.activation_fcn_name.lower() in ['relu', 'leakyrelu', 'prelu']:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                elif self.activation_fcn_name.lower() in ['tanh', 'sigmoid']:
                    nn.init.xavier_uniform_(layer.weight)

                if layer.bias is not None:
                    # Initialize bias to zero if present
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # Perform forward pass iterating through all layers of DNN
        for layer in self.layers:
            x = layer(x)
        return x 


# %% Image normalization classes
class NormalizeImg(nn.Module):
    def __init__(self, normaliz_value : float = 255.0):
        super(NormalizeImg, self).__init__()
        self.normaliz_value = normaliz_value

    def forward(self, x):
        return x / self.normaliz_value  # Normalize to [0, 1]

# Define the ReNormalize layer
class ReNormalizeImg(nn.Module):
    def __init__(self, normaliz_value: float = 255.0):
        super(ReNormalizeImg, self).__init__()
        self.normaliz_value = normaliz_value

    def forward(self, x):
        return x * self.normaliz_value  # Re-normalize to [0, 255]


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


def ValidateDictLoading(model: nn.Module | nn.ModuleDict | nn.ModuleList, modelName: str, filepath: str):

    # Load the saved state dict (just to compare)
    checkpoint = torch.load(os.path.join(filepath, modelName+'.pth'))
    saved_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Get the current state dict from the model
    current_state_dict = model.state_dict()

    # Check if the model's parameters match the saved parameters
    for param_name in current_state_dict:
        if not torch.equal(current_state_dict[param_name], saved_state_dict[param_name]):
            raise ValueError(f"Mismatch found in parameter: {param_name}")

    else:
        print("All model parameters are correctly loaded.")

