# Module to apply activation functions in forward pass instead of defining them in the model class
from typing import Union
import torch
from pyTorchAutoForge.api.torch import * 

# For initialization

#############################################################################################################################################


class torchModel(torch.nn.Module):
    '''Custom base class inheriting nn.Module to define a PyTorch NN model, augmented with saving/loading routines like Pytorch Lightning.'''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def saveCheckpoint(self):
        SaveTorchModel() # To do: first input must be the model itself (traced or not)


#############################################################################################################################################
