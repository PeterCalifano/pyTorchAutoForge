from dataclasses import dataclass
import torch
from torch import nn
from typing import Literal
import torch.nn.functional as F
import numpy as np


from dataclasses import dataclass
import torch
from torch import nn
from typing import Literal
import torch.nn.functional as F
import numpy as np

# TODO implement an optional "residual connection" feature

from pyTorchAutoForge.model_building.factories.block_factories import activ_types, pooling_types, regularizer_types, init_methods, pooling_types_1d, pooling_types_2d, pooling_types_3d

# TODO review and test, should be an ONNx compatible AdaptiveAvgPool2d
class CustomAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size : tuple[int,int] | int):
        super(CustomAdaptiveAvgPool2d, self).__init__()
        
        # Output_size: (H_out, W_out) or int
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        self.output_size = tuple(output_size)
        if self.output_size[0] == 1 and self.output_size[1] == 1:
            self.forward = self.forward_impl_GAP2d
        else:
            self.forward = self.forward_impl_adaptive2d

    # Define overloads for different input dimensions
    def forward_impl_GAP2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward_impl_GAP2d _summary_

        _extended_summary_

        :param x: _description_
        :type x: torch.Tensor
        :return: _description_
        :rtype: torch.Tensor
        """
        # For output size 1x1, i.e. Global Average Pooling
        return x.mean(dim=(2, 3), keepdim=True)

    def forward_impl_adaptive2d(self, x: torch.Tensor) -> torch.Tensor:
        # shape: (N, C, H_in, W_in) -> (N, C, H_out, W_out)
        return F.interpolate(x, size=self.output_size, mode="area")

    #def forward(self, x: torch.Tensor) -> torch.Tensor:

class CustomAdaptiveMaxPool2d(nn.Module):
    def __init__(self, output_size : tuple[int,int] | int):
        super(CustomAdaptiveMaxPool2d, self).__init__()

        # Output_size: (H_out, W_out) or int
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        self.output_size = tuple(output_size)

        if self.output_size[0] == 1 and self.output_size[1] == 1:
            self.forward = self.forward_impl_GMP2d
        else:
            self.forward = self.forward_impl_adaptive2d

    # Define overloads for different input dimensions
    def forward_impl_GMP2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward_impl_GMP2d _summary_

        _extended_summary_

        :param x: _description_
        :type x: torch.Tensor
        :return: _description_
        :rtype: torch.Tensor
        """
        # For output size 1x1, i.e. Global Max Pooling
        return x.max(dim=(2, 3), keepdim=True)[0]

    def forward_impl_adaptive2d(self, x: torch.Tensor) -> torch.Tensor:
        
        # Shape: (N, C, H_in, W_in) -> (N, C, H_out, W_out)
        H_out, W_out = self.output_size

        B, C, H, W = x.shape
        if H % H_out != 0 or W % W_out != 0:
            raise ValueError(
                f"ONNX-safe adaptive max pool needs input divisible by output; "
                f"got H={H}, W={W}, target={self.output_size}"
            )
        x = x.reshape(B, C, H_out, H // H_out, W_out, W // W_out)
        return x.amax(dim=(3, 5))
