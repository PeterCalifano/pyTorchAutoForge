'''Script to test the loss visualization method suggested by Li et al. (2018) in the paper "Visualizing the Loss Landscape of Neural Nets"
Using the code from the paper's GitHub repository: '''

# Import modules
import sys, os

# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import PyTorch.pyTorchAutoForge.pyTorchAutoForge.pyTorchAutoForge as pyTorchAutoForge # Custom torch tools
import limbPixelExtraction_CNN_NN  # Custom model classes
import datasetPreparation
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


def GetRandomDirection(state_dict):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    return [torch.randn(w.size()) for k, w in state_dict.items()] # Note: here list comprehension is used

def ComputeRandomDisplacementsForModel(model):
    '''Function to compute filter normalized directions of random displacements to get loss function values'''

    # Compute first random directions from model state dictionary
    randomDir1 = GetRandomDirection(model.state_dict())
    
    # Compute second random directions from model state dictionary
    randomDir2 = GetRandomDirection(model.state_dict())



    # Apply filter normalization to random directions
    for d1, d2, weight in zip(randomDir1, randomDir2, model.parameters()):
        
        wNorm  = weight.view((weight.shape[0],-1)).norm(dim=(1),keepdim=True)[:,:,None,None]
        dirNorm1 = d1.view((d1.shape[0],-1)).norm(dim=(1),keepdim=True)[:,:,None,None]
        dirNorm2 = d2.view((d2.shape[0],-1)).norm(dim=(1),keepdim=True)[:,:,None,None]

        d1.data = d1.cuda() * (wNorm/(dirNorm1.cuda()+1e-10))
        d2.data = d2.cuda() * (wNorm/(dirNorm2.cuda()+1e-10))


    return filterNormalizedDir1, filterNormalizedDir2



def main():
    
    print('------------------------------- TEST: Loss functions classes -------------------------------')

    LOSS_TYPE = 4 # 0: Conic + L2, # 1: Conic + L2 + Quadratic OutOfPatch, # 2: Normalized Conic + L2 + OutOfPatch, 
                  # 3: Polar-n-direction distance + OutOfPatch, #4: MSE + OutOfPatch + ConicLoss
    # Loss function parameters
    params = {'ConicLossWeightCoeff': 0, 'RectExpWeightCoeff': 0}

    lossFcn = pyTorchAutoForge.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_NormalizedConicLossWithMSEandOutOfPatch_asTensor, params)
    #model = ModelClasses.HorizonExtractionEnhancerCNNv3maxDeeper

    tracedModelSavePath = '/home/peterc/devDir/MachineLearning_PeterCdev/checkpoints/HorizonPixCorrector_CNNv3max_largerCNNdeeperNN_run0003'
    tracedModelName = 'HorizonPixCorrector_CNNv3max_largerCNNdeeperNN_run0003_0002_cuda0.pt'


    # Load torch traced model from file
    torchWrapper = pyTorchAutoForge.TorchModel_MATLABwrap(tracedModelName, tracedModelSavePath)

    modelParams = torchWrapper.trainedModel.parameters()





    

if __name__ == '__main__':
    main()