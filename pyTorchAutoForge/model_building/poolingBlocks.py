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
    def __init__(self, output_size):
        super(CustomAdaptiveAvgPool2d, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward _summary_

        _extended_summary_

        :param x: _description_
        :type x: torch.Tensor
        :return: _description_
        :rtype: torch.Tensor
        """
    
        shape_x = x.shape

        if (shape_x[-1] < self.output_size[-1]):

            paddzero = torch.zeros(
                (shape_x[0], shape_x[1], shape_x[2], self.output_size[-1] - shape_x[-1]))
            
            paddzero = paddzero.to(x.device)
            x = torch.cat((x, paddzero), axis=-1)

        # Compute kernel and stride sizes to reach desired output size
        stride_size = np.floor(
            np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        
        kernel_size = np.array(x.shape[-2:]) - \
            (self.output_size - 1) * stride_size
        
        avg = nn.AvgPool2d(kernel_size=list(kernel_size),
                           stride=list(stride_size))
        
        x = avg(x)

        return x
