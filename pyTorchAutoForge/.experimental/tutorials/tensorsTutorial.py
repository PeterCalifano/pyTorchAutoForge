# Example script from video 1 of sentdex series on PyTorch - 01/08/2023 - by PC
# Reference: https://www.youtube.com/watch?v=BzcBsTou0C0&t=950s

# Import modules 
import numpy as np
import matplotlib.pyplot as plt
import torch # Pytorch


# About the (simplified) reason why GPUs makes DL much faster: each operation involving (weights, biases) pairs, that are the "neuronal connections"
# moving from the Input Layers to the Output Layers through the Inner (Hidden, read-only) Layers, is elementary and small. However, as the 
# number of neurons and parameters is increased, the number of operations grows exponentially. A GPU is ideal to execute a very large number
# of elementary operations in parallel, whereas CPUs do not. 
# %%
# Define basic classes of Pytorch (tensors, namely multi-dimensional arrays)
xtensor = torch.Tensor([2, 1])
ytensor = torch.Tensor([3, 5, 7, 6])

print("xtensor object:", xtensor)
print("xtensor dtype: ", xtensor.type) 
print("ytensor: ", ytensor)

# Typical methods of numpy are identically transported in pytorch acting as methods of class "tensor"
zerosTensor = torch.zeros([5, 5, 6])
print("Example of zeros 3D array: ", zerosTensor)

# Reshape is different though, for instance: torch.view(__self__, [size1, size2, ... , sizeN])
# ACHTUNG: In NN in general, multi-dims arrays must be flattened  (i.e., making them column vectors)
reshaped_y = ytensor.view(2, 2)
print('Reshaped y tensor to flattened array: ', reshaped_y)