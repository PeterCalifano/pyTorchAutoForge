import torch.nn as nn
from typing import Union
import inspect


def build_activation_layer(activation_name, show_defaults: bool = False, *args, **kwargs) -> nn.Module:
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

    # Inspect the activation class to get parameter defaults
    sig = inspect.signature(activation_class)
    required_args = set()
    optional_args = {}

    for param in sig.parameters.values():
        if param.name != 'inplace' and param.name != 'self' and param.name != 'args' and param.name != 'kwargs':
            if param.default == param.empty and param.name != 'self':
                required_args.add(param.name)
            elif param.default != param.empty:
                optional_args[param.name] = param.default

    # Print optional arguments with defaults if requested
    if show_defaults:
        defaults_to_show = {k: v for k,
                            v in optional_args.items() if k not in kwargs}
        if defaults_to_show:
            defaults_info = ", ".join(
                f"{key}={value}" for key, value in defaults_to_show.items())
            print(f"Optional defaults for {activation_name}: {defaults_info}")

    # Check for missing required args
    missing_args = required_args - kwargs.keys()
    if missing_args:
        raise ValueError(
            f"Missing required arguments for {activation_name}: {missing_args}"
        )

    # Return the activation instance
    return activation_class(*args, **kwargs)


# TODO
class ConvolutionalLayer():
    def __init__(self, dict_key, *args, **kwargs) -> nn.Module:
        pass


def build_normalization_layer(dict_key, show_defaults: bool = False, *args, **kwargs) -> nn.Module:
    # Define normalization layers using a regular dictionary with class references
    normalization_layers = {
        'BatchNorm2d': nn.BatchNorm2d,
        'LayerNorm': nn.LayerNorm,
        'InstanceNorm2d': nn.InstanceNorm2d,
        'GroupNorm': nn.GroupNorm
    }

    if dict_key not in normalization_layers:
        raise ValueError(f"Unknown normalization type: {dict_key}")

    # Retrieve the normalization class
    normalization_class = normalization_layers[dict_key]

    # Inspect the class to get parameter defaults
    sig = inspect.signature(normalization_class)
    required_args = set()
    optional_args = {}

    for param in sig.parameters.values():
        if param.default == param.empty and param.name != 'self':
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

    # Instantiate and return the normalization layer with the validated arguments
    return normalization_class(*args, **kwargs)
