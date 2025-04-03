import torch.nn as nn
from typing import Union, Any, Dict, List, Optional, Tuple
import inspect
from enum import Enum
from dataclasses import dataclass, field


# DEVNOTE: as anticipated (and suggested by Giacomo) the current generic approach to build functions (without defaults) is a bit cumbersome to use. 

# %% Configuration classes

class block_type(Enum):
    conv = 0
    linear = 1

#class layer_type(Enum):
#    conv = 0
#    norm = 1
#    activation = 2
#    pooling = 3
#    linear = 4


# Base classes for layers and blocks
@dataclass 
class LayerConfig: # Empty class to force inheritance
    pass

@dataclass
class BlockConfig:
    type: block_type           # Block type
    layers: List[LayerConfig]  # List of layers in the block


@dataclass
class ModelConfig:
    modules: List[Union[LayerConfig, BlockConfig]]  # List of layers and blocks in the model

# Single layer configuration classes
@dataclass
class ConvLayerConfig(LayerConfig):
    in_channels: int
    out_channels: int
    typename: str = "conv2d"
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1


@dataclass
class NormLayerConfig(LayerConfig):
    num_features: int
    typename: str = "BatchNorm2d"

@dataclass
class ActivationLayerConfig(LayerConfig):
    typename: str = "PRelu"
    inplace: bool = True  # Only for ReLU-like activations
    leaky_slope: float = 0.01  # Only for LeakyReLU

@dataclass
class PoolingLayerConfig(LayerConfig):
    typename: str = "MaxPool2d"
    kernel_size: int = 2
    stride: Optional[int] = 2
    padding: int = 0

@dataclass
class LinearLayerConfig(LayerConfig):
    in_features: int
    out_features: int
    typename: str = "Linear"




# %% Single layer builders
def validate_args(layer_class: nn.Module, show_defaults: bool, dict_key: str, *args, **kwargs):
    """
    Validates the arguments for a given layer class by inspecting its signature.

    Parameters:
    layer_class (nn.Module): The neural network layer class to validate arguments for.
    show_defaults (bool): If True, prints the optional arguments and their default values that are not provided by the user.
    dict_key (str): A string key used for identifying the layer in error messages.
    *args: Additional positional arguments (not used in this function).
    **kwargs: Keyword arguments provided by the user.

    Raises:
    ValueError: If any required arguments are missing from kwargs.

    Notes:
    - This function inspects the signature of the provided layer_class to determine which arguments are required and which are optional.
    - If show_defaults is True, it prints the optional arguments and their default values that are not provided by the user.
    - It raises a ValueError if any required arguments are missing from kwargs.
    - It includes default values for optional arguments if they are not provided in kwargs.
    """
    # Inspect the class to get parameter defaults
    sig = inspect.signature(layer_class)
    required_args = set()
    optional_args = {}

    for param in sig.parameters.values():
        if param.default == param.empty and param.name != 'self' and param.name != 'args' and param.name != 'kwargs':
            required_args.add(param.name)
        elif param.default != param.empty:
            optional_args[param.name] = param.default

    # Filter optional args to only show those not provided by the user
    if show_defaults:
        defaults_to_show = {k: v for k,
                            v in optional_args.items() if k not in kwargs}
        if defaults_to_show:
            defaults_info = ", ".join(
                f"{key}={value}" for key, value in defaults_to_show.items())
            print(f"Optional defaults for {dict_key}: {defaults_info}")

    # Check for missing required args
    missing_args = required_args - kwargs.keys()
    if missing_args:
        raise ValueError(
            f"Missing required arguments for {dict_key}: {missing_args}")

    # Include defaults for optional args if they’re not provided in kwargs
    for key, default_value in optional_args.items():
        kwargs.setdefault(key, default_value)

    return kwargs

def build_activation_layer(activation_name : str, show_defaults: bool = False, *args, **kwargs) -> nn.Module:
    """
    Factory function to build and return a PyTorch activation layer based on the provided activation name.

    Args:
        activation_name (str): The name of the activation function to build. 
                               Supported values are 'relu', 'lrelu', 'prelu', 'sigmoid', and 'tanh'.
        show_defaults (bool, optional): If True, prints the default values of optional arguments for the activation function. Defaults to False.
        *args: Additional positional arguments to pass to the activation function.
        **kwargs: Additional keyword arguments to pass to the activation function.

    Raises:
        ValueError: If the provided activation_name is not supported.
        ValueError: If required arguments for the activation function are missing.

    Returns:
        nn.Module: An instance of the specified activation function.
    """

    activation_name = activation_name.lower()

    # Define activations with required and optional arguments
    activations = {
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU,
        'prelu': nn.PReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }

    if activation_name not in activations:
        raise ValueError(f"Unknown activation type: {activation_name}")

    # Retrieve the activation class
    activation_class = activations[activation_name]

    # Perform argument validation
    kwargs = validate_args(activation_class, show_defaults,
                           dict_key=activation_name, *args, **kwargs)

    # Return the activation instance
    return activation_class(*args, **kwargs)


def build_convolutional_layer(convolution_name:str='conv2d', show_defaults: bool = False, *args, **kwargs) -> nn.Module:
    """
    Builds a convolutional layer based on the specified type and arguments.
    Args:
        convolution_name (str): The type of convolutional layer to build. 
                                Options are 'conv1d', 'conv2d', and 'conv3d'. 
                                Default is 'conv2d'.
        show_defaults (bool): If True, display the default arguments for the 
                              specified convolutional layer. Default is False.
        *args: Variable length argument list to pass to the convolutional layer.
        **kwargs: Arbitrary keyword arguments to pass to the convolutional layer.
    Returns:
        nn.Module: An instance of the specified convolutional layer.
    Raises:
        ValueError: If an unknown convolution type is specified.
    """
    
    convolution_name = convolution_name.lower()

    # Define convolutional layers using a regular dictionary with class references
    convolution_layers = {
        'conv1d': nn.Conv1d,
        'conv2d': nn.Conv2d,
        'conv3d': nn.Conv3d
    }

    if convolution_name not in convolution_layers:
        raise ValueError(f"Unknown convolution type: {convolution_name}")

    # Retrieve the convolution class
    convolution_class = convolution_layers[convolution_name]

    # Perform argument validation
    kwargs = validate_args(convolution_class, show_defaults,
                  dict_key=convolution_name, *args, **kwargs)

    # Instantiate and return the convolution layer with the validated arguments
    return convolution_class(*args, **kwargs)


def build_normalization_layer(normalization_name: str = 'groupnorm', show_defaults: bool = False, *args, **kwargs) -> nn.Module:
    """
    Builds and returns a normalization layer based on the specified normalization type.

    Args:
        normalization_name (str): The name of the normalization layer to build. 
                                  Options include 'batchnorm2d', 'layernorm', 'instancenorm2d', and 'groupnorm'.
                                  Default is 'groupnorm'.
        show_defaults (bool): If True, displays the default arguments for the specified normalization layer.
                              Default is False.
        *args: Additional positional arguments to pass to the normalization layer constructor.
        **kwargs: Additional keyword arguments to pass to the normalization layer constructor.

    Returns:
        nn.Module: An instance of the specified normalization layer.

    Raises:
        ValueError: If the specified normalization_name is not recognized.

    Example:
        >>> norm_layer = build_normalization_layer('batchnorm2d', num_features=64)
        >>> print(norm_layer)
        BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    """

    normalization_name = normalization_name.lower()

    # Define normalization layers using a regular dictionary with class references
    normalization_layers = {
        'batchnorm2d': nn.BatchNorm2d,
        'layernorm': nn.LayerNorm,
        'instancenorm2d': nn.InstanceNorm2d,
        'groupnorm': nn.GroupNorm
    }

    if normalization_name not in normalization_layers:
        raise ValueError(f"Unknown normalization type: {normalization_name}")

    # Retrieve the normalization class
    normalization_class = normalization_layers[normalization_name]

    # Perform argument validation
    kwargs = validate_args(normalization_class, show_defaults, dict_key=normalization_name, *args, **kwargs)

    # Instantiate and return the normalization layer with the validated arguments
    return normalization_class(*args, **kwargs)


def build_pooling_layer(pooling_name, show_defaults: bool = False, *args, **kwargs) -> nn.Module:
    # Define pooling layers with both standard and adaptive pooling classes
    pooling_layers = {
        'MaxPool1d': nn.MaxPool1d,
        'MaxPool2d': nn.MaxPool2d,
        'MaxPool3d': nn.MaxPool3d,
        'AvgPool1d': nn.AvgPool1d,
        'AvgPool2d': nn.AvgPool2d,
        'AvgPool3d': nn.AvgPool3d,
        'AdaptiveMaxPool1d': nn.AdaptiveMaxPool1d,
        'AdaptiveMaxPool2d': nn.AdaptiveMaxPool2d,
        'AdaptiveMaxPool3d': nn.AdaptiveMaxPool3d,
        'AdaptiveAvgPool1d': nn.AdaptiveAvgPool1d,
        'AdaptiveAvgPool2d': nn.AdaptiveAvgPool2d,
        'AdaptiveAvgPool3d': nn.AdaptiveAvgPool3d
    }

    if pooling_name not in pooling_layers:
        raise ValueError(f"Unknown pooling type: {pooling_name}")

    # Retrieve the pooling class
    pooling_class = pooling_layers[pooling_name]

    # Perform argument validation
    kwargs = validate_args(pooling_class, show_defaults, dict_key=pooling_name, *args, **kwargs)

    # Instantiate and return the pooling layer with validated arguments
    return pooling_class(*args, **kwargs)


def build_linear_layer(linear_name, show_defaults: bool = False, *args, **kwargs) -> nn.Module:
    # Define linear layers using a dictionary with class references
    linear_layers = {
        'Linear': nn.Linear,
        'Bilinear': nn.Bilinear,
        'LazyLinear': nn.LazyLinear
    }

    if linear_name not in linear_layers:
        raise ValueError(f"Unknown linear layer type: {linear_name}")

    # Retrieve the linear class
    linear_class = linear_layers[linear_name]

    # Perform argument validation
    kwargs = validate_args(linear_class, show_defaults, dict_key=linear_name, *args, **kwargs)
    
    # Instantiate and return the linear layer with validated arguments
    return linear_class(*args, **kwargs)


# %% Block builders
#def build_convolutional_block(block_config: BlockConfig) -> nn.Module:
#    pass
#def build_linear_block(block_config: BlockConfig) -> nn.Module:
#    pass

# NOTE: there seems to be no necessity of block specific builders.

# %% Builder wrapper function
#def build_model(ModelConfig):


# Function to build the generic layer from configuration
def build_layer(config: LayerConfig) -> nn.Module:

    assert (isinstance(config, LayerConfig)), f"Input object type {type(config)} not an instance of LayerConfig or its subclasses"

    # Unpack parameters from the configuration
    layer_type = config.typename
    params = vars(config)  # Convert the dataclass fields to a dictionary

    # Remove the 'type' key as it's not an actual parameter for the builder functions
    params.pop('typename')

    if isinstance(config, ConvLayerConfig):
        return build_convolutional_layer(layer_type, **params)

    elif isinstance(config, NormLayerConfig):
        return build_normalization_layer(layer_type, **params)

    elif isinstance(config, ActivationLayerConfig):
        return build_activation_layer(layer_type, **params)

    elif isinstance(config, PoolingLayerConfig):
        return build_pooling_layer(layer_type, **params)

    elif isinstance(config, LinearLayerConfig):
        return build_linear_layer(layer_type, **params)

    else:
        raise ValueError(f"Unsupported configuration type: {type(config)}")


# Function to build the generic block from configuration
def build_block(block_config: BlockConfig) -> nn.Module:
    """
    Builds a block from a BlockConfig instance by sequentially creating layers.
    """

    assert(isinstance(block_config, BlockConfig)), f"Input object type {type(block_config)} not an instance of BlockConfig"

    layers = [build_layer(layer_config) for layer_config in block_config.layers]
    return nn.Sequential(*layers)



