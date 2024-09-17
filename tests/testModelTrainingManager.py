# Import modules
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
import numpy as np
from typing import Union

import PyTorch.pyTorchAutoForge.pyTorchAutoForge.pyTorchAutoForge as pyTorchAutoForge

from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim
import torch.nn.functional as F # Module to apply activation functions in forward pass instead of defining them in the model class


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