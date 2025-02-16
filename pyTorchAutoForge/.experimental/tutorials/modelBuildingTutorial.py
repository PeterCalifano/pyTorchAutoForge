# Script implementing the PyTorch "Building Models with PyTorch" example as exercise by PeterC - 14-05-2024
# Reference: https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html

# Import modules
import torch
from torch import nn
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
import datetime
import numpy as np

# %% BASIC CONCEPTS TO KNOW
# The basic components to build models in PyTorch: 
# 1) torch.nn.Module     --> class encapsulating all PyTorch Neural Networks and layers models
# 2) torch.nn.Parameters --> class encapsulating all weights, biases and parameters of layers. Learnable weights of nn.Module are instances of nn.Parameters. 
#                            NOTE: this is a sub-class of torch.Tensor, which have special behaviour when attribute of nn.Module. 
#                            Accessible using parameters() method.

# Note that when calling parameters() to check the layer weights, the second object of tensor tells whether the autograd is "registering" the gradients for the layer

# %% NETWORK LAYERS TYPES
# 1) LINEAR (FULLY CONNECTED) --> Just one matrix multiplication (weight matrix * input vector) and one matrix addition (+ bias vector)
size_in = 10 # A random number
size_out = 15

linerLayer = nn.Linear(size_in, size_out)

# 2) CONVOLUTIONAL 

# EXAMPLE: LeNet model
import torch.functional as F

class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = torch.nn.Conv2d(1, 6, 5) # Take gray images (first input = 1, since 1 channel), 
                                              # 2nd input --> how many features (i.e. convolutional kernels), 3rd input: size of the kernel
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling operating locally on a window of [2,2] pixels.
                                                        # This merges the image content locally --> halves the size of the image. N
        # If the size is a square you can specify just a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
# 3) RECURRENT (LSTM and GRU): Long-Short Term Memory and Gated Recurrent Units
# EXAMPLE of LSTM:

class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim) 
        # Embedding layer mapping from one input feature space to the output feature space. The inputs for layer definition are the dimensions.

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim) # LSTM model in PyTorch

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size) # Classifier layer learning the map between a given point in the 
                                                                   # embedding space (of size hidden_dim) to the class space (with size equal to the number of classes)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores    
    
# 4) Transformers
# Torch has a class "Transformers"
# torch.nn.Transformer, encapsulating (TransformerEncoder, TransformerDecoder) and subcomponents (TransformerEncoderLayer, TransformerDecoderLayer)

# %% DATA MANIPULATION TYPES
# Layers that are non trainable (no trainable parameters), but performs some operations on the data
# 1) MAX or MIN POOLING --> merge cells assigning the local max or min value to the output (single scalar value)
#    Example: max_pool(5, 5) checks the max over the (5,5) window and gives it as output. Effectively works like max_value = max(tensor)
# 2) NORMALIZATION --> recentre and normalize a tensor (i.e. makes it zero mean and white). 
#    This is typically beneficial for activation functions and to improve behaviour against vanishing gradients and learning rates
# 3) DROPOUT --> randomly set a value in a fraction of the input tensor. This forces a sparse representation and are only active at TRAINING TIME.
#    Disabled at INFERENCE TIME. That's one of the layers affected by the methods .train() or .eval().
# 4) ACTIVATION FUNCTIONS --> this is trule what makes a NN a non-linear universal function approximator.
# 5) LOSS FUNCTIONS --> determines what the networks learns and how it is trained.