# Script created by PeterC 17-05-2024 reproducing model presentend in Pugliatti's PhD thesis (C4, Table 4.1)
# Implemented in PyTorch. Datasets on Zenodo: https://zenodo.org/records/7107409

# Import modules
import torch
from torch import nn
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
import torch.nn.functional as torchFunc
import datetime


# %% MODEL ARCHITECTURE (CNN ENCODER HEAD FOR U-NET)

class EncoderForUnet_SemanticSegment(nn.Module):
    def __init__(self, numOfInputChannels, numOfOutputChannels, kernelSize=3, poolingSize=2) -> None:
        super().__init__()

        # Layers attributes
        self.numOfInputChannels  = [1, 16, 32, 64, 128, 256*16, 256] # Number of input channels (3rd dimension if 1 and 2 are the image height and width)
        self.numOfOutputChannels = [16, 32, 64, 128, 256, 256, 128] # Number of output channels resulting from kernel convolution
        self.kernelSize = 3
        self.poolingSize = 2
        self.outputLayerSize = 7
        self.alphaLeaky = 0.3

        # Model architecture
        self.conv2dL1 = nn.Conv2d(numOfInputChannels[0], numOfOutputChannels[0], kernelSize) 
        self.maxPoolL1 = nn.MaxPool2d(poolingSize, poolingSize) # Note: all MaxPooling are [2,2] since the size of the image halves in Mattia's table

        self.conv2dL2 = nn.Conv2d(numOfInputChannels[1], numOfOutputChannels[1], kernelSize) 
        self.maxPoolL2 = nn.MaxPool2d(poolingSize, poolingSize) 

        self.conv2dL3 = nn.Conv2d(numOfInputChannels[2], numOfOutputChannels[2], kernelSize) 
        self.maxPoolL3 = nn.MaxPool2d(poolingSize, poolingSize) 

        self.conv2dL4 = nn.Conv2d(numOfInputChannels[3], numOfOutputChannels[3], kernelSize) 
        self.maxPoolL4 = nn.MaxPool2d(poolingSize, poolingSize) 

        self.conv2dL5 = nn.Conv2d(numOfInputChannels[4], numOfOutputChannels[4], kernelSize) 
        self.maxPoolL5 = nn.MaxPool2d(poolingSize, poolingSize)

        self.dropoutL6 = nn.Dropout2d(0.2)
        self.FlattenL6 = nn.Flatten()

        self.DenseL7 = nn.Linear(numOfInputChannels[5], numOfOutputChannels[5])
        self.dropoutL7 = nn.Dropout1d(0.2)

        self.DenseL8 = nn.Linear(numOfInputChannels[6], numOfOutputChannels[6])
        self.DenseOutput = nn.Linear(numOfOutputChannels[6], 7)

    def forward(self, inputSample):
        # Convolutional layers
        # L1 (Input)
        val = self.maxPoolL1(torchFunc.leaky_relu(self.conv2dL1(inputSample), self.alphaLeaky))
        # L2
        val = self.maxPoolL2(torchFunc.leaky_relu(self.conv2dL2(val), self.alphaLeaky))
        # L3
        val = self.maxPoolL3(torchFunc.leaky_relu(self.conv2dL3(val), self.alphaLeaky))
        # L4
        val = self.maxPoolL4(torchFunc.leaky_relu(self.conv2dL4(val), self.alphaLeaky))
        # L5
        val = self.maxPoolL5(torchFunc.leaky_relu(self.conv2dL5(val), self.alphaLeaky))

        # Fully connected layers
        # L6 Dropout and Flatten
        val = self.dropoutL6(val)
        val = self.FlattenL6(val) # Flatten data to get input to Fully Connected layers
        # L7 Linear
        val = self.DenseL7(val)
        # L8 Linear 
        val = self.DenseL8(val)
        # Output layer
        featuresVec = self.DenseOutput(val)
        return featuresVec
    
# Suggested Training parameters: Adam, 200 batch size, Loss metric: SCCE, Accuracy metric: accuracy, 100 epochs