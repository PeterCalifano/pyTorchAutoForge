
# Import modules
import sys, os
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import PyTorch.pyTorchAutoForge.pyTorchAutoForge.pyTorchAutoForge as pyTorchAutoForge  # Custom torch tools
import limbPixelExtraction_CNN_NN

import torch
import datetime
from torch import nn
from scipy.spatial.transform import Rotation

from torch.utils.data import Dataset
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
from typing import Union
import numpy as np

from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim


def main():

    # SETTINGS and PARAMETERS 
    batch_size = 16*2 # Defines batch size in dataset
    TRAINING_PERC = 0.80
    #outChannelsSizes = [16, 32, 75, 15] 
    outChannelsSizes = [16, 32, 75, 15] 
    kernelSizes = [3, 1]
    learnRate = 1E-10
    momentumValue = 0.001
    optimizerID = 1 # 0
    device = pyTorchAutoForge.GetDevice()
    exportTracedModel = True

    # MODEL CLASS TYPE
    classID = 0

    if classID == 0:
        modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNN
    elif classID == 1:
        modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv2

    modelCNN_NN = modelClass(outChannelsSizes, kernelSizes)
    print(modelCNN_NN)

if __name__ == '__main__':
    main()