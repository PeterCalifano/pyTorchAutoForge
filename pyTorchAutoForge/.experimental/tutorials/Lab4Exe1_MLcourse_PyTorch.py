# Example script from article on NN basics in PyTorch - 16/04/2023 - by PC
# Reference: https://towardsdatascience.com/neural-networks-forward-pass-and-backpropagation-be3b75a1cfcc
# TensorFlow Keras version in Lab4Exe1.py script

# Example adapted from:
# [1] Brunton, S. L., & Kutz, J. N. (2022).
# Data-driven science and engineering: Machine learning,
# dynamical systems, and control. Cambridge University Press.
# Data from [1].
# 

# %%
# Import modules 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import io
import os

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.utils import to_categorical

# PyTorch
import torch # Pytorch
import torch.nn as nntorch
from torch.autograd import Variable

# %% 
# Example of NN in PyTorch
inFeatureDim = 1
outFeatureDim = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Set device to run the NN model

# NOTE: differently from tf the sizes of inputs and outputs of each layer are both specified, which makes more explicit the network architecture
NNmodel = nntorch.Sequential(
    nntorch.Linear(inFeatureDim, outFeatureDim), # INPUT LAYER --> Defines how the layers are combined
    nntorch.ReLU(), # Specify activation function of Input layer (NOTE: this is largely different from Keras)
    nntorch.Linear(2,2), # HIDDEN LAYER
    nntorch.ReLU(), # Activation of hidden layer
    nntorch.Linear(2,1) # OUTPUT LAYER
).to(device)

# Loss function definition
lossFcn = nntorch.MSELoss()

# Initialize optimizer (Stochastic Gradient Descent)
n_epochs = 2
learning_rate = 1e-2
optL = nntorch.optim.SGD(NNmodel.parameters(), lr=learning_rate) 

optL.zero_grad()

# How to get weights or biases from the architecture
# In PyTorch the activation functions are layers themselves --> (w,b) only in even numbered layers
wLayer0 = NNmodel[0].weight.detach().to(device).numpy()
bLayer0 = NNmodel[0].bias.detach().to(device).numpy()

# Gradients computation
gradw_L0 = NNmodel[0].weight.grad.detach().to(device).numpy()
gradb_L0 = NNmodel[0].bias.grad.detach().to(device).numpy()

# EXAMPLE OF TRAINING ITERATION
# Definition of input and labels
inputVal = 0
LabelVal = 1

t1_u = Variable(torch.from_numpy(inputVal).float(), requires_grad=True).to(device)
t1_c = Variable(torch.from_numpy(LabelVal).float(), requires_grad=True).to(device)

t_ucol = torch.tensor(t1_u, device=device, requires_grad=True)
t_ccol = torch.tensor(t1_c, device=device, requires_grad=True)

# Flattening to 1D vector is done as follows:
t_u1 = torch.cat([t_ucol.view((-1, 1))], dim=1)
t_c1 = torch.cat([t_ccol.view((-1, 1))], dim=1)

# INFERENCE (FORWARD STEP)
outputVal = NNmodel(t_u1)

# Cost function evaluation
trainingLabel = t_c1
lossTraining = lossFcn(outputVal, trainingLabel)

# BACKPROPAGATION STEP
lossTraining.backward(retain_graph=True)
# Optimizer step is called by the following method:
optL.step()
