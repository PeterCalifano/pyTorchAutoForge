'''
 Test model forward method input reshaping and dataloader class, created by PeterC - 24-06-2024
'''

# Import modules
import sys, os
# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import PyTorch.pyTorchAutoForge.pyTorchAutoForge.pyTorchAutoForge as pyTorchAutoForge  # Custom torch tools

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
import limbPixelExtraction_CNN_NN


def main():

    # SETTINGS and PARAMETERS 
    batch_size = 16*2 # Defines batch size in dataset
    TRAINING_PERC = 0.85
    #outChannelsSizes = [16, 32, 75, 15] 
    outChannelsSizes = [256, 128, 75, 50] 
    kernelSizes = [3, 3]
    learnRate = 1E-9
    momentumValue = 0.001

    modelClass = limbPixelExtraction_CNN_NN.HorizonExtractionEnhancerCNNv1max
    modelCNN_NN = modelClass(outChannelsSizes, kernelSizes)

    # %% TORCH DATASET LOADING
    pathToDataset = '/home/peterc/devDir/MachineLearning_PeterCdev/checkpoints/HorizonPixCorrector_CNNv1max_largerCNN_run3sampleDatasetToONNx'
    dataset = pyTorchAutoForge.LoadTorchDataset(pathToDataset)

    datasetSize = len(dataset)
    datasetLoader  = DataLoader(dataset, 2, shuffle=True)

    print(datasetLoader)
    
    
    # %% TORCH MODEL LOADING
    
    modelCNN_test = modelClass(outChannelsSizes, kernelSizes)


    # %% MODEL FORWARD METHOD TESTING
    inputSamples = pyTorchAutoForge.GetSamplesFromDataset(datasetLoader, numOfSamples=1)    

    lossFcn = pyTorchAutoForge.CustomLossFcn(limbPixelExtraction_CNN_NN.MoonLimbPixConvEnhancer_NormalizedLossFcnWithOutOfPatchTerm)

    
    examplePrediction, exampleLosses, inputSampleList = pyTorchAutoForge.EvaluateModel(datasetLoader, modelCNN_test.to(pyTorchAutoForge.GetDevice()), lossFcn)

    print('Example prediction:', examplePrediction)
    print('Example input list:', inputSampleList)


if __name__ == '__main__':
    main()