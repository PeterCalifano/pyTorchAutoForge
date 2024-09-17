# Import modules
import sys, os

import PyTorch.pyTorchAutoForge.pyTorchAutoForge.pyTorchAutoForge
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))

import PyTorch.pyTorchAutoForge.pyTorchAutoForge.pyTorchAutoForge as pyTorchAutoForge # Custom torch tools
from sklearn import preprocessing # Import scikit-learn for dataset preparation

import torch
import datetime, json
from torch import nn
from scipy.spatial.transform import Rotation

from torch.utils.data import Dataset
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

import numpy as np

from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

class LossLandscapePlotter():
    def __init__(self, model:nn.Module, lossFcn:nn.Module, dataloader:DataLoader, device=pyTorchAutoForge.GetDevice()):

        self.model = model
        self.lossFcn = lossFcn    
        self.dataloader = dataloader
        self.device = device

    def GetRandomDirection(self):
        '''Function to generate a random direction for the loss landscape plot'''
        direction = []
        for param in self.model.parameters():
            if param.requires_grad:
                direction.append(torch.randn_like(param).to(param.device))
        
        return direction

    def NormalizeDirections(self):
        '''Function to normalize the directions to have the same norm as the model parameters'''
        norm_d1, norm_d2 = [], []

        for p1, p2, param in zip(self.d1, self.d2, self.model.parameters()):

            if param.requires_grad:
                norm = torch.norm(param)
                norm_d1.append(p1 / torch.norm(p1) * norm)
                norm_d2.append(p2 / torch.norm(p2) * norm)

        return norm_d1, norm_d2

    def AddDirectionToModel(self, model, direction, alpha):
        '''Function to add a direction to the model parameters'''
        index = 0
        for param in model.parameters():
            if param.requires_grad:
                param.data.add_(direction[index], alpha=alpha)
                index += 1

    # TODO: modify this function to use CustomLossFcn class
    def CalculateLoss(self):
        '''
        Function to calculate the loss of the model on the dataset.
        NOTE: this function assumes that lossFcn returns the average loss function over the batch
        '''
        self.model.eval() # Set model to evaluation mode
        loss = 0.0
        Nsamples = 0

        with torch.no_grad(): # Tell torch that gradients are not required
            
            for X, Y in self.dataLoader:

                batchSize = X.size(0)
                Nsamples += batchSize

                X, Y = X.to(self.device), Y.to(self.device)
                # Evaluate model
                outputs = self.model(X)
                # Evaluate loss and sum to total count
                loss += batchSize * self.lossFn(outputs, Y).item()

        return loss / Nsamples # Return the average loss function value for the entire dataset

    def PlotLossLandscape(self, dataLoader, lossFn, device, xRange=(-1, 1), yRange=(-1, 1), stepSize=0.1):
        '''Function to plot the loss landscape of a model using normalized random directions'''

        # Generate random directions
        self.d1 = self.GetRandomDirection()
        self.d2 = self.GetRandomDirection()
        # Normalize directions
        d1, d2 = self.NormalizeDirections()

        # Create meshgrid and loss values to plot
        xVals = np.arange(xRange[0], xRange[1], stepSize)
        yVals = np.arange(yRange[0], yRange[1], stepSize)
        losses = np.zeros((len(xVals), len(yVals)))

        originalState = [param.clone() for param in self.model.parameters()]
        # Plot data generation loops
        for i, x in enumerate(xVals):
            for j, y in enumerate(yVals):
                # Add directions to model parameters
                self.AddDirectionToModel(self.model, d1, x)
                self.AddDirectionToModel(self.model, d2, y)
                # Evaluate loss at new point
                losses[i, j] = self.CalculateLoss()
                # Reset model parameters
                for param, originalParam in zip(self.model.parameters(), originalState):
                    param.data.copy_(originalParam) # Copy the original state back to the model parameters

        # Create contour plot data
        X, Y = np.meshgrid(xVals, yVals)
        Z = losses.T  # Transpose to match the correct orientation

        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Direction 1')
        plt.ylabel('Direction 2')
        plt.title('Loss Landscape')
        plt.show()

# Example usage
# Assuming you have a dataloader `dataLoader` and loss function `lossFn`
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# PlotLossLandscape(mofrl, dataLoader, lossFn, device)

def main():
    
    print('------------------------------- TEST: Loss functions classes -------------------------------')

    LOSS_TYPE = 4 # 0: Conic + L2, # 1: Conic + L2 + Quadratic OutOfPatch, # 2: Normalized Conic + L2 + OutOfPatch, 
                  # 3: Polar-n-direction distance + OutOfPatch, #4: MSE + OutOfPatch + ConicLoss
    # Loss function parameters
    params = {'ConicLossWeightCoeff': 0, 'RectExpWeightCoeff': 0}

    lossFcn = pyTorchAutoForge.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_NormalizedConicLossWithMSEandOutOfPatch_asTensor, params)
    model = ModelClasses.HorizonExtractionEnhancerCNNv3maxDeeper


if __name__ == '__main__':
    main()