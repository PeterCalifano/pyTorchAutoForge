# Import modules
import torch, sys, os
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
import numpy as np
from typing import Union

sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))

import PyTorch.pyTorchAutoForge.pyTorchAutoForge.pyTorchAutoForge as pyTorchAutoForge

from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim
import torch.nn.functional as F # Module to apply activation functions in forward pass instead of defining them in the model class


class GenericNN(nn.Module):
    def __init__(self, config, device=pyTorchAutoForge.GetDevice()):
        super(GenericNN, self).__init__()

        self.config = config

        # Call autobuilder to construct model from configuration
        self.model = NetworkAutoBuilder(config).to(device)

        # Get forward method from autobuilder
        self.forward = self.model.forward

# Network auto builder class from configuration dict proposed by GPT - 08-07-2024
class NetworkAutoBuilder(nn.Module):

    def __init__(self, config):
        super(NetworkAutoBuilder, self).__init__()

        self.config = config
        self.layerNames = []

        self.buildModel() # Build the model from the configuration dictionary

    def buildModel(self):   
        '''Function to build the model from the configuration dictionary'''
        for idx, layerConfig in enumerate(self.config['layers']):

            layerType = layerConfig['type'].lower() # Get the layer type from the configuration
            layerName = f"layer_{idx}_{layerType}" # Get the layer name from the configuration
            self.layerNames.append(layerName)

            if layerType == 'linear':
                self.addLinearLayer(layerConfig, layerName)
            elif layerType == 'conv2d':
                self.addConv2dLayer(layerConfig, layerName)
            elif layerType == 'maxpool2d':
                self.addMaxPool2dLayer(layerConfig, layerName)
            elif layerType == 'flatten':
                self.addFlattenLayer(layerName)
            elif layerType == 'dropout':
                self.addDropoutLayer(layerConfig, layerName)
            elif layerType == 'batchnorm2d':
                self.addBatchNorm2dLayer(layerConfig, layerName)
            else:
                raise ValueError(f"Layer type {layerType} not found in supported list.")

    # AUXILIARY FUNCTIONS TO BUILD MODEL
    def addLinearLayer(self, layerConfig, layerName):
        '''Function to add a Linear layer to the model'''

        inFeatures = layerConfig['in_features']
        outFeatures = layerConfig['out_features']
        bias = layerConfig.get('bias', True)

        layer = nn.Linear(inFeatures, outFeatures, bias=bias)
        setattr(self, layerName, layer)

        # Add activation layer if present
        activation = layerConfig.get('activation', None)
        if activation:
            self.addActivationLayer(activation, f"{layerName}_activation")


    def addConv2dLayer(self, layerConfig, layerName):
        '''Function to add a Conv2d layer to the model'''

        inChannels = layerConfig['in_channels']
        outChannels = layerConfig['out_channels']
        kernelSize = layerConfig['kernel_size']

        stride = layerConfig.get('stride', 1) # Get the stride from the configuration, if not present, default to 1
        padding = layerConfig.get('padding', 0)
        bias = layerConfig.get('bias', True)

        layer = nn.Conv2d(inChannels, outChannels, kernelSize, stride=stride, padding=padding, bias=bias)
        setattr(self, layerName, layer)

        # Add activation layer if present
        activation = layerConfig.get('activation', None)
        if activation:
            self.addActivationLayer(activation, f"{layerName}_activation")


    def addMaxPool2dLayer(self, layerConfig, layerName):
        '''Function to add a MaxPool2d layer to the model'''

        kernelSize = layerConfig.get('kernel_size', 2)
        stride = layerConfig.get('stride', None)
        padding = layerConfig.get('padding', 0)

        layer = nn.MaxPool2d(kernelSize, stride=stride, padding=padding)
        setattr(self, layerName, layer)

    def addFlattenLayer(self, layerName):
        '''Function to add a Flatten layer to the model'''
        layer = nn.Flatten()
        setattr(self, layerName, layer)

    def addDropoutLayer(self, layerConfig, layerName):
        '''Function to add a Dropout layer to the model'''
        prob = layerConfig['p']
        layer = nn.Dropout(prob)
        setattr(self, layerName, layer)

    def addBatchNorm2dLayer(self, layerConfig, layerName):
        '''Function to add a BatchNorm2d layer to the model'''

        idx = self.config['layers'].index(layerConfig) # Not sure this works: TEST
        numFeatures = layerConfig.get('num_features', self.config['layers'][idx-1]['out_features'])
        
        eps=layerConfig.get('eps', 1e-05)
        momentum=layerConfig.get('momentum', 0.1)

        layer = nn.BatchNorm2d(numFeatures, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        setattr(self, layerName, layer)


    def addActivationLayer(self, activation, layerName, activParams=None):
        if activParams is None:
            activParams = {}

        '''Function to add an activation layer to the model'''

        if activation.lower() == 'relu':
            activationLayer = nn.ReLU()

        elif activation.lower() == 'sigmoid':
            activationLayer = nn.Sigmoid()

        elif activation.lower() == 'tanh':
            activationLayer = nn.Tanh()

        elif activation.lower() == 'softmax':
            activationLayer = nn.Softmax(dim=1)

        elif activation.lower() == 'leakyrelu':
            activationLayer = nn.LeakyReLU(negative_slope=activParams.get('negative_slope', 0.01))

        elif activation.lower() == 'elu':
            activationLayer = nn.ELU()

        elif activation.lower() == 'selu':
            activationLayer = nn.SELU()

        elif activation.lower() == 'prelu':  
            # NOT clear: does prelu for dense layers assign one learnable parameter per node?
            # Paper and documentation talk about one learnable parameter per channel 
                   
            num_parameters = activParams.get('num_parameters', 1)
            activationLayer = nn.PReLU(num_parameters=num_parameters)

        else:
            raise ValueError(f"Activation function {activation} not found in supported list.")
        
        if activationLayer:
            setattr(self, layerName, activationLayer)
    
    # FORWARD METHOD
    def forward(self, x):
        '''Function performing the forward pass of the model'''

        for layerConfig in self.config['layers']:
            # Get layer name from configuration dictionary
            layerName = layerConfig.get('name')
            # Perform forward pass through all layers
            x = getattr(self, layerName)(x)

        return x


    def summary(self):
        '''Function to print a summary of the model architecture'''
        print("Model Summary:")
        for idL, layerConfig in enumerate(self.config['layers']):

            layerName = self.layerNames[idL]
            layer = getattr(self, layerName)

            print(f"Layer: {layerName}, Type: {type(layer).__name__}, Configuration: {layer}")


def main():
    print('------------------------------- DEVTEST: Model autobuilder from configuration input  -------------------------------')

    # Example usage:
    config = {
        'layers': [
            {'type': 'Conv2d', 'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'name': 'conv1'},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'name': 'pool1'},
            {'type': 'Conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'name': 'conv2'},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'name': 'pool2'},
            {'type': 'Flatten', 'name': 'flatten'},
            {'type': 'Linear', 'in_features': 64*5*5, 'out_features': 128, 'name': 'fc1'}, # Adjust in_features according to input shape
            {'type': 'Dropout', 'p': 0.5, 'name': 'dropout1'},
            {'type': 'Linear', 'in_features': 128, 'out_features': 10, 'name': 'output'}
            ],   
    }

    model = NetworkAutoBuilder(config)
    model.summary()


if __name__ == '__main__':
    main()