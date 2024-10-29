import torch.nn as nn
from typing import Union
import inspect


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

def build_activation_layer(activation_name : str, show_defaults: bool = False, *args, **kwargs) -> nn.Module:
    """
    Build and return a PyTorch activation layer based on the provided activation name.

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

    normalization_layers = normalization_layers.lower()

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
