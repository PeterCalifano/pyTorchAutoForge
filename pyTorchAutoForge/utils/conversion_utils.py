from torch import Tensor, from_numpy
import numpy as np
from numpy import ndarray
import torch

# Interfaces between numpy and torch tensors
def torch_to_numpy(tensor: Tensor | ndarray, dtype : type = np.float32) -> ndarray:

    if isinstance(tensor, Tensor):
        # Convert to torch tensor to numpy array
        return (tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()).astype(dtype)

    elif isinstance(tensor, ndarray):
        # Return the array itself
        return tensor
    else:
        raise ValueError("Input must be a torch.Tensor or np.ndarray")


def numpy_to_torch(array: Tensor | ndarray, dtype : torch.dtype = torch.float32) -> Tensor:

    if isinstance(array, ndarray):
        # Convert numpy array to torch tensor
        return from_numpy(array).to(torch.float32)

    elif isinstance(array, Tensor):
        # Return the tensor itself
        return array
    else:
        raise ValueError("Input must be a torch.Tensor or np.ndarray")
