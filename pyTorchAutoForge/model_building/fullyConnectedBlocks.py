import torch
from torch import nn
from typing import Literal


# TODO implement an optional "residual connection" feature
class FullyConnectedBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activ_type: Literal["prelu", "sigmoid",
                                     "relu", "tanh", "none"] = "sigmoid",
                 regularizer_type: Literal["dropout",
                                           "batchnorm", "groupnorm", "none"] = "none",
                 regularizer_param: float | int = 0.0,
                 prelu_params: Literal["all", "unique"] = "unique",
                 init_method: Literal["xavier_uniform",
                                      "kaiming_uniform",
                                      "xavier_normal",
                                      "kaiming_normal",
                                      "orthogonal"] = "xavier_uniform",
                 **kwargs
                 ):
        super().__init__()

        # Define linear layer
        self.linear = nn.Linear(in_channels, out_channels)

        # Activation selection
        self.activ: nn.Module | nn.Identity = nn.Identity()
        match activ_type.lower():
            case "prelu":
                num_p = out_channels if prelu_params == "all" else 1
                self.activ = nn.PReLU(num_p)

            case "relu": self.activ = nn.ReLU()
            case "sigmoid": self.activ = nn.Sigmoid()
            case "tanh": self.activ = nn.Tanh()
            case "none": self.activ = nn.Identity()
            case _: raise ValueError(f"Unsupported activation: {activ_type}")

        # Regularizer selection
        self.regularizer: nn.Module | nn.Identity = nn.Identity()
        match regularizer_type.lower():
            case "dropout":
                assert 0 < regularizer_param < 1
                self.regularizer = nn.Dropout(regularizer_param)
            case "batchnorm": self.regularizer = nn.BatchNorm1d(out_channels)
            case "groupnorm":
                # DOUBT Group norm for fully connected layers?
                assert regularizer_param > 0
                self.regularizer = nn.GroupNorm(
                    int(regularizer_param), out_channels)

            case "none": self.regularizer = nn.Identity()
            case _: raise ValueError(f"Unsupported regularizer: {regularizer_type}")

        # Initialize weights using specified method
        self._initialize_weights(init_method)

    def _initialize_weights(self,
                            init_method_type: Literal["xavier_uniform",
                                                      "kaiming_uniform",
                                                      "xavier_normal",
                                                      "kaiming_normal",
                                                      "orthogonal"] = "xavier_uniform"):
        """
        Initializes the weights of the linear layer using the specified initialization method.

        Args:
            init_method_type (str): The initialization method to use. 
            One of "xavier_uniform", "kaiming_uniform", "xavier_normal", 
            "kaiming_normal", or "orthogonal".
        """

        match init_method_type.lower():
            case "xavier_uniform": nn.init.xavier_uniform_(self.linear.weight)
            case "kaiming_uniform": nn.init.kaiming_uniform_(self.linear.weight)
            case "xavier_normal": nn.init.xavier_normal_(self.linear.weight)
            case "kaiming_normal": nn.init.kaiming_normal_(self.linear.weight)
            case "orthogonal": nn.init.orthogonal_(self.linear.weight)
            case _: raise ValueError(f"Unsupported initialization method: {init_method_type}")

        # Initialize bias to zero
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):

        x = self.linear(x)
        x = self.activ(x)
        x = self.regularizer(x)

        return x
