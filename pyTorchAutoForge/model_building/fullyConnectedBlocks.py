import torch
from torch import nn
from typing import Literal
from dataclasses import dataclass
import dataclasses

from pyTorchAutoForge.model_building.factories.block_factories import _initialize_fcnblock_weights, _activation_factory, regularizer_types, activ_types, init_methods, _regularizer_factory

# %% FullyConnectedBlocks module
@dataclass
class FullyConnectedBlockConfig:
    in_channels: int
    out_channels: int
    activ_type: activ_types = "sigmoid"
    regularizer_type: regularizer_types = "none"
    regularizer_param: float | int = 0.0
    prelu_params: Literal["all", "unique"] = "unique"
    init_method: init_methods = "xavier_normal"

class FullyConnectedBlock(nn.Module):
    """
    FullyConnectedBlock summary.

    This class implements a fully connected block using PyTorch modules. It consists
    of a linear layer followed by an activation function and a regularization layer (optionally).

    Attributes:
        linear (torch.nn.Linear): Linear transformation layer mapping in_channels to out_channels.
        activ (torch.nn.Module | torch.nn.Identity): Non-linear activation function.
        regularizer (torch.nn.Module | torch.nn.Identity): Regularization layer applied after activation.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activ_type: activ_types = "sigmoid",
                 regularizer_type: regularizer_types = "none",
                 regularizer_param: float | int = 0.0,
                 prelu_params: Literal["all", "unique"] = "unique",
                 init_method_type: init_methods = "xavier_normal",
                 **kwargs
                 ):
        super().__init__()

        # Save input/output dimensions for interfaces specification
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define linear layer
        self.linear = nn.Linear(in_features=in_channels,
                                out_features=out_channels,
                                bias=True)

        # Activation selection
        self.activ: nn.Module | nn.Identity = nn.Identity()
        self.activ = _activation_factory(activ_type=activ_type,
                                         out_channels=out_channels,
                                         prelu_params=prelu_params)

        # Regularizer selection
        # Regularizer selection
        self.regularizer: nn.Module | nn.Identity = nn.Identity()
        self.regularizer = _regularizer_factory(ndims = 1,
                                                regularizer_type=regularizer_type,
                                                regularizer_param=regularizer_param,
                                                out_channels=out_channels
                                                )

        # Initialize bias to zero
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        # Call layers weights init
        self.__initialize_weights__(init_method_type)

    def __initialize_weights__(self, init_method_type: Literal["xavier_uniform",
                                                        "kaiming_uniform",
                                                        "xavier_normal",
                                                        "kaiming_normal",
                                                        "orthogonal"] = "xavier_uniform"):
        """
        Initializes the weights of the linear layer using the specified initialization method.
        """
        _initialize_fcnblock_weights(self, init_method_type=init_method_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear(x)
        x = self.activ(x)
        x: torch.Tensor = self.regularizer(x)

        return x

# %% FullyConnectedBlocksStack modules
class FullyConnectedBlocksStack(nn.Module):
    """
    FullyConnectedBlocksStack summary.

    This class implements a stack of fully connected blocks, allowing for sequential
    addition of FullyConnectedBlock modules to construct a neural network architecture.

    Attributes:
        blocks (torch.nn.ModuleList): A list of FullyConnectedBlock instances forming the stack.
    """
    def __init__(self, block_cfg_list: list[FullyConnectedBlockConfig] | None = None):
        super().__init__()
        self.blocks = nn.ModuleList()

        if block_cfg_list is not None:
            for cfg in block_cfg_list:
                self.add_block(cfg)

    def add_block(self, block_cfg: FullyConnectedBlockConfig):
        """
        Adds a fully connected block to the block stack.

        Args:
            block_cfg (FullyConnectedBlockConfig): A configuration object containing parameters
            to create a FullyConnectedBlock.
        """

        block = FullyConnectedBlock(**dataclasses.asdict(block_cfg))
        self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

class FullyConnectedResidualBlockStack(FullyConnectedBlocksStack):
    """
    FullyConnectedResidualBlockStack implements a stack of fully connected blocks with residual connections.

    This class extends FullyConnectedBlocksStack by enabling residual connections between selected input
    features and output features. The features from the input tensor (specified by input_skip_indices) are
    carried forward and added to the corresponding output features (specified by output_skip_indices) after
    passing through all fully connected blocks.

    Args:
        input_skip_indices (list of int): List of indices from the input tensor to be used for the residual connection.
        output_skip_indices (list of int): List of indices in the output tensor where the residual connection will be applied.
        block_cfg_list (list[FullyConnectedBlockConfig], optional): List of configuration objects for the fully connected blocks.
            If provided, the stack is built by sequentially adding blocks based on these configurations.

    Raises:
        TypeError: If input_skip_indices or output_skip_indices are not lists of integers.
        ValueError: If block_cfg_list is empty or if any skip index exceeds the number of features in the input or output tensor.
    """
    def __init__(self, 
                 input_skip_indices : list[int],
                 output_skip_indices: list[int], 
                 block_cfg_list: list[FullyConnectedBlockConfig] | None = None):
        super().__init__(block_cfg_list=None)

        if not isinstance(input_skip_indices, list):
            raise TypeError("input_skip_indices must be a list of indices.")
        if not all(isinstance(i, int) for i in input_skip_indices):
            raise TypeError("All elements in input_skip_indices must be integers.")

        if not isinstance(output_skip_indices, list):
            raise TypeError("output_skip_indices must be a list of indices.")
        if not all(isinstance(i, int) for i in output_skip_indices):
            raise TypeError("All elements in output_skip_indices must be integers.")

        # Store skip indices as params
        self.register_buffer("input_skip_indices",
                             torch.tensor(input_skip_indices))
        self.register_buffer("output_skip_indices",
                             torch.tensor(output_skip_indices))
        
        if block_cfg_list is not None:
            for cfg in block_cfg_list:
                self.add_block(cfg)
                
            self._check_input_output_skip_indices()

    def _check_input_output_skip_indices(self):
        # Check if input_skip_indices and output_skip_indices have valid entries (not larger than input and output dimensions)
        
        # Check at least one block is present
        if len(self.blocks) == 0:
            raise ValueError("No blocks defined in the stack. Please add at least one block.")

        # All input indices must be less than the number of features in the input tensor
        if self._buffers["input_skip_indices"].max().item() >= self.blocks[0].linear.in_features:
            raise ValueError("input_skip_indices must be less than the number of input features.")
        # All output indices must be less than the number of features in the output tensor
        if self._buffers["output_skip_indices"].max().item() >= self.blocks[-1].linear.out_features:
            raise ValueError("output_skip_indices must be less than the number of output features.")

        # Number of indices must be <= number of features in the input and output tensors
        if self._buffers["input_skip_indices"].numel() > self.blocks[0].linear.in_features:
            raise ValueError("input_skip_indices length exceeds number of input features.")
        if self._buffers["output_skip_indices"].numel() > self.blocks[-1].linear.out_features:
            raise ValueError("output_skip_indices length exceeds number of output features.")

    def add_block(self, block_cfg: FullyConnectedBlockConfig):
        """
        Adds a fully connected block and verifies residual connection indices.

        Adds a fully connected block using the provided configuration, then checks that
        the input and output skip indices remain valid for the current architecture of the
        residual block stack.

        Args:
            block_cfg (FullyConnectedBlockConfig): Configuration for the fully connected block.
        """
        super().add_block(block_cfg)
        self._check_input_output_skip_indices()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Extract skip values along the feature dimension (assumes x shape is [batch, features])
        residual_connect : torch.Tensor = torch.index_select(x, 1, self.input_skip_indices)
        
        for block in self.blocks:
            x = block(x)
        
        # Compute updated values for the specified output indices
        out_plus_skip = x[:, self.output_skip_indices] + residual_connect

        # Replace the values at those indices using index_copy (differentiable).
        x = x.index_copy(1, self.output_skip_indices, out_plus_skip)

        return x
