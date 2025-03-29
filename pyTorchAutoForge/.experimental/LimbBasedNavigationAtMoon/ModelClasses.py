'''
Script created by PeterC 30-06-2024 to group all model classes for project: Limb-based navigation using CNN-NN network
Reference forum discussion: https://discuss.pytorch.org/t/adding-input-layer-after-a-hidden-layer/29225

# BRIEF DESCRIPTION
The network takes windows of Moon images where the illuminated limb is present, which may be a feature map already identifying the limb if edge detection has been performed.
Convolutional layers process the image patch to extract features, then stacked with several other inputs downstream. Among these, the relative attitude of the camera wrt to the target body 
and the Sun direction in the image plane. A series of fully connected layers then map the flatten 1D vector of features plus the contextual information to infer a correction of the edge pixels.
This correction is intended to remove the effect of the imaged body morphology (i.e., the network will account for the DEM "backward") such that the refined edge pixels better adhere to the
assumption of ellipsoidal/spherical body without terrain effects. The extracted pixels are then used by the Christian Robinson algorithm to perform derivation of the position vector.
Variants using fully connected layers only, without 2D patch information are included as well.
'''

import torch, sys, os
from limbPixelExtraction_CNN_NN import *

import datetime
from torch import nn
from math import sqrt
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils

from typing import Union
import numpy as np

from torch.utils.tensorboard import SummaryWriter 
import torch.optim as optim
import torch.nn.functional as torchFunc # Module to apply activation functions in forward pass instead of defining them in the model class

# For initialization
import torch.nn.init as init

# %% Custom training function for Moon limb pixel extraction enhancer V3maxDeeper (with target average radius in pixels) - 30-06-2024
'''
Architecture characteristics: Conv. layers, max pooling, deeper fully connected layers, dropout, tanh and leaky ReLU activation
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates, target average radius in pixels.
'''
class HorizonExtractionEnhancer_CNNv3maxDeeper(nn.Module):
        ############################################################################################################    
    def __init__(self, outChannelsSizes:list, kernelSizes, useBatchNorm = False, poolingKernelSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2
        self.useBatchNorm = useBatchNorm

        if useBatchNorm:
            print('Batch normalization layers: enabled.')

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)

        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = convBlockOutputSize[1] 
        
        self.LinearInputSkipSize = 8 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.maxPoolL2 = nn.MaxPool2d(poolingKernelSize, 1) 
        self.batchNormL2 = nn.BatchNorm2d(self.outChannelsSizes[1], eps=1E-5, momentum=0.1, affine=True)

        # Fully Connected predictor
        self.FlattenL3 = nn.Flatten()

        self.dropoutL4 = nn.Dropout(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=False)
        self.batchNormL4 = nn.BatchNorm1d(self.outChannelsSizes[2], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)
        self.batchNormL5 = nn.BatchNorm1d(self.outChannelsSizes[3], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL6 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL6 = nn.Linear(self.outChannelsSizes[3], self.outChannelsSizes[4], bias=True)
        self.batchNormL6 = nn.BatchNorm1d(self.outChannelsSizes[4], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        # Output layer
        #self.batchNormL7 = nn.BatchNorm1d(self.outChannelsSizes[4], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        self.DenseOutput = nn.Linear(self.outChannelsSizes[4], 2, bias=True)

        # Initialize weights of layers
        self.__initialize_weights__()

        ############################################################################################################
    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''
        # Leaky_ReLU activation layers
        init.kaiming_uniform_(self.conv2dL1.weight, nonlinearity='leaky_relu') 
        init.constant_(self.conv2dL1.bias, 0)

        init.kaiming_uniform_(self.conv2dL2.weight, nonlinearity='leaky_relu') 
        init.constant_(self.conv2dL2.bias, 0)

        init.kaiming_uniform_(self.DenseL6.weight, nonlinearity='leaky_relu') 
        init.constant_(self.DenseL6.bias, 0)

        # Tanh activation layers
        init.xavier_uniform_(self.DenseL4.weight) 
        init.xavier_uniform_(self.DenseL5.weight) 
        init.constant_(self.DenseL5.bias, 0)

        # TODO: make initi automatic and not structure specific
        #for module in self.modules():
        #    print('Module:', module)

        ############################################################################################################
    def forward(self, inputSample):
        '''Forward prediction method'''
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        
        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # Convolutional layers
        # L1 (Input)
        val = self.maxPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))

        # L2
        val = self.conv2dL2(val)
        if self.useBatchNorm:
            val = self.batchNormL2(val)
        val = self.maxPoolL2(torchFunc.leaky_relu(val, self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        #val = self.batchNormL3(val)
        val = self.DenseL4(val)
        val = self.dropoutL4(val)

        if self.useBatchNorm:
            val = self.batchNormL4(val)
        val = torchFunc.tanh(val)

        # L5
        val = self.DenseL5(val)
        val = self.dropoutL5(val)

        if self.useBatchNorm:
            val = self.batchNormL5(val)
        val = torchFunc.tanh(val)

        # L6
        val = self.DenseL6(val)
        val = self.dropoutL6(val)

        if self.useBatchNorm:
            val = self.batchNormL6(val)
        val = torchFunc.leaky_relu(val, self.alphaLeaky)

        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    
#############################################################################################################################################
# %% Custom training function for Moon limb pixel extraction enhancer FullyConnected NNv4 (with target average radius in pixels) - 30-06-2024
class HorizonExtractionEnhancer_FullyConnNNv4iter(nn.Module):
    def __init__(self, outChannelsSizes:list, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2

        self.LinearInputSize = 8 # Should be 10 after modification
        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Fully Connected predictor
        self.DenseL1 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[0], bias=False)

        self.dropoutL2 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL2 = nn.Linear(self.outChannelsSizes[0], self.outChannelsSizes[1], bias=True)
        self.BatchNormL2 = nn.BatchNorm1d(self.outChannelsSizes[1], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL3 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL3 = nn.Linear(self.outChannelsSizes[1], self.outChannelsSizes[2], bias=True)
        self.BatchNormL3 = nn.BatchNorm1d(self.outChannelsSizes[2], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.DenseL4 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)
        self.BatchNormL4 = nn.BatchNorm1d(self.outChannelsSizes[3], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        # Auto building for loop: EXPERIMENTAL
        #for idL in range():

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=False)

    def forward(self, inputSample):
        # Get inputs that are not image pixels from input samples
        contextualInfoInput = inputSample[:, self.imagePixSize:] 

        # Fully Connected Layers
        # L1 (Input layer)
        val = torchFunc.prelu(self.DenseL1(contextualInfoInput)) 
        # L2
        val = self.dropoutL2(val)
        val = torchFunc.prelu(self.DenseL2(val))
        # L3
        val = self.dropoutL3(val)
        val = torchFunc.prelu(self.DenseL3(val))
        # L4
        val = torchFunc.prelu(self.DenseL4(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    
#############################################################################################################################################

#############################################################################################################################################
# %% Custom training function for Moon limb pixel extraction enhancer ResNet-like V5maxDeeper (with target average radius in pixels) - 01-07-2024
class HorizonExtractionEnhancer_ResCNNv5maxDeeper(nn.Module):
    ############################################################################################################
    def __init__(self, outChannelsSizes:list, kernelSizes, useBatchNorm = False, poolingKernelSize=2, alphaDropCoeff=0.3, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2
        self.useBatchNorm = useBatchNorm

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)

        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = convBlockOutputSize[1] 
        
        self.LinearInputSkipSize = 6 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.maxPoolL2 = nn.MaxPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here?
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=False)

        self.batchNormL5 = nn.BatchNorm1d(self.outChannelsSizes[2], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)
        
        self.batchNormL6 = nn.BatchNorm1d(self.outChannelsSizes[3], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        self.dropoutL6 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL6 = nn.Linear(self.outChannelsSizes[3], self.outChannelsSizes[4], bias=True)

        # Output layer
        self.batchNormL7 = nn.BatchNorm1d(self.outChannelsSizes[4], eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        self.DensePreOutput = nn.Linear(self.outChannelsSizes[4]+2, 2, bias=True)  # PATCH CENTRE SKIP CONNECTION
        self.DenseOutput = nn.Linear(2, 2, bias=True)

    ############################################################################################################
    def forward(self, inputSample):
        '''Forward prediction method'''
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        
        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        
        device = img2Dinput.device
        contextualInfoInput = torch.tensor([inputSample[:, self.imagePixSize:12], inputSample[:, 14]], dtype=torch.float32, device=device)
        pixPatchCentre = inputSample[:, 12:14]

        # Convolutional layers
        # L1 (Input)
        val = self.maxPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))
        # L2
        val = self.maxPoolL2(torchFunc.leaky_relu(self.conv2dL2(val), self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.tanh(self.DenseL4(val))

        # L5
        if self.useBatchNorm:
            val = self.batchNormL5(val)
        val = self.dropoutL5(val)
        val = torchFunc.tanh(self.DenseL5(val))

        # L6
        if self.useBatchNorm:
            val = self.batchNormL6(val)
        val = self.dropoutL6(val)
        val = torchFunc.leaky_relu(self.DenseL6(val), self.alphaLeaky)

        # Output layer
        if self.useBatchNorm:
            val = self.batchNormL7(val)
    
        # Add pixel Patch centre coordinates
        val = self.DensePreOutput(val)
        val = val + pixPatchCentre
        correctedPix = self.DenseOutput(val)

        return correctedPix
    
#############################################################################################################################################
class HorizonExtractionEnhancer_ShortCNNv6maxDeeper(nn.Module):
    '''Experimental model with semi-automatic initialization. Structure of the architecture is fixed, but hyperparameters of this are specified at instantiation using
    the "parametersConfig" dictionary with the following keys: kernelSizes, useBatchNorm, poolingKernelSize, alphaDropCoeff, alphaLeaky, patchSize, LinearInputSkipSize'''
    def __init__(self, outChannelsSizes:list, parametersConfig) -> None:
        super().__init__()

        # Extract all the inputs of the class init method from dictionary parametersConfig, else use default values

        kernelSizes = parametersConfig.get('kernelSizes', [5])
        useBatchNorm = parametersConfig.get('useBatchNorm', False)
        poolingKernelSize = parametersConfig.get('poolingKernelSize', 2)
        alphaDropCoeff = parametersConfig.get('alphaDropCoeff', 0.1)
        alphaLeaky = parametersConfig.get('alphaLeaky', 0.01)
        patchSize = parametersConfig.get('patchSize', 7)
            
        assert(len(outChannelsSizes) == 4) # 7 layers in the model
        assert(len(kernelSizes) == 1)
        assert('LinearInputSkipSize' in parametersConfig.keys())

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 1
        self.useBatchNorm = useBatchNorm

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)

        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = convBlockOutputSize[1] # convBlockOutputSize is tuple ((imgWidth, imgHeight), flattenedSize*nOutFeatures)
    
        self.LinearInputSkipSize = parametersConfig['LinearInputSkipSize'] #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        #self.alphaLeaky = alphaLeaky

        # Model architecture
        idLayer = 0
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[idLayer], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingKernelSize, 1)
        self.preluL1 = nn.PReLU(self.outChannelsSizes[idLayer])
        idLayer += 1

        #self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        #self.maxPoolL2 = nn.MaxPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        self.FlattenL2 = nn.Flatten()
        
        self.dropoutL3 = nn.Dropout(alphaDropCoeff)
        self.DenseL3 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[idLayer], bias=False)
        self.batchNormL3 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL3 = nn.PReLU()

        idLayer += 1

        self.dropoutL4 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], bias=True)
        self.batchNormL4 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL4 = nn.PReLU()

        idLayer += 1

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], bias=True)
        self.batchNormL5 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL5 = nn.PReLU()

        idLayer += 1

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[idLayer-1], 2, bias=True)

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''
        # ReLU activation layers
        init.kaiming_uniform_(self.conv2dL1.weight, nonlinearity='leaky_relu') 
        init.constant_(self.conv2dL1.bias, 0)

        init.kaiming_uniform_(self.DenseL5.weight, nonlinearity='leaky_relu') 
        init.constant_(self.DenseL5.bias, 0)

        init.kaiming_uniform_(self.DenseL3.weight, nonlinearity='leaky_relu') 
        init.kaiming_uniform_(self.DenseL4.weight, nonlinearity='leaky_relu')  
        init.constant_(self.DenseL4.bias, 0)

    def forward(self, inputSample):
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        
        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # Convolutional layers
        # L1 (Input)
        val = self.maxPoolL1( torchFunc.prelu(self.conv2dL1(img2Dinput), self.preluL1.weight) )

        # Fully Connected Layers
        # L2
        val = self.FlattenL2(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L3 
        #val = self.batchNormL3(val)
        val = self.DenseL3(val)
        val = self.dropoutL3(val)
        if self.useBatchNorm:
            val = self.batchNormL3(val)
        val = torchFunc.prelu(val, self.preluL3.weight)

        # L4
        val = self.DenseL4(val)
        val = self.dropoutL4(val)
        if self.useBatchNorm:
            val = self.batchNormL4(val)
        val = torchFunc.prelu(val, self.preluL4.weight)

        # L5
        val = self.DenseL5(val)
        val = self.dropoutL5(val)
        if self.useBatchNorm:
            val = self.batchNormL5(val)
        val = torchFunc.prelu(val, self.preluL5.weight)

        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection


# %% Horizon Extraction Enhanced Fully Connected only, with Conic parameters from initial guess - 08-07-2024
class HorizonExtractionEnhancer_deepNNv8(nn.Module):
    '''Horizon Extraction Enhanced Fully Connected only, with Conic parameters from initial guess. Parametric ReLU activations for all layers.
    Dropout and batch normalization for all layers. Fixed layers number with parametric sizes.'''
    def __init__(self, parametersConfig) -> None:
        super().__init__()

        # Extract all the inputs of the class init method from dictionary parametersConfig, else use default values
        useBatchNorm = parametersConfig.get('useBatchNorm', False)
        alphaDropCoeff = parametersConfig.get('alphaDropCoeff', 0.1)
        self.LinearInputSize = parametersConfig.get('LinearInputSize', 58)

        # Model parameters
        self.outChannelsSizes = parametersConfig['outChannelsSizes']
        self.useBatchNorm = useBatchNorm


        # Model architecture

        idLayer = 0
        self.DenseL1 = nn.Linear(self.LinearInputSize, self.outChannelsSizes[idLayer], bias=False) 
        self.batchNormL1 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL1 = nn.PReLU(self.outChannelsSizes[idLayer])
        idLayer += 1

        self.dropoutL2 = nn.Dropout(alphaDropCoeff)
        self.DenseL2 = nn.Linear(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], bias=True) 
        self.batchNormL2 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL2 = nn.PReLU(self.outChannelsSizes[idLayer])
        idLayer += 1

        self.dropoutL3 = nn.Dropout(alphaDropCoeff)
        self.DenseL3 = nn.Linear(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], bias=True)
        self.batchNormL3 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL3 = nn.PReLU()
        idLayer += 1

        self.dropoutL4 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], bias=True)
        self.batchNormL4 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL4 = nn.PReLU()
        idLayer += 1

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], bias=True)
        self.batchNormL5 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL5 = nn.PReLU()
        idLayer += 1

        self.dropoutL6 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL6 = nn.Linear(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], bias=True)
        self.batchNormL6 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL6 = nn.PReLU()

        idLayer += 1
        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[idLayer-1], 2, bias=True)

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''
       
        # ReLU activation layers
        init.kaiming_uniform_(self.DenseL1.weight, nonlinearity='leaky_relu') 

        init.kaiming_uniform_(self.DenseL2.weight, nonlinearity='leaky_relu') 
        init.constant_(self.DenseL2.bias, 0)

        init.kaiming_uniform_(self.DenseL3.weight, nonlinearity='leaky_relu') 
        init.constant_(self.DenseL3.bias, 0)

        init.kaiming_uniform_(self.DenseL4.weight, nonlinearity='leaky_relu') 
        init.constant_(self.DenseL4.bias, 0)

        init.kaiming_uniform_(self.DenseL5.weight, nonlinearity='leaky_relu') 
        init.constant_(self.DenseL5.bias, 0)

        init.kaiming_uniform_(self.DenseL6.weight, nonlinearity='leaky_relu') 
        init.constant_(self.DenseL6.bias, 0)


    def forward(self, inputSample):
            
        # Fully Connected Layers
        # L1
        val = self.DenseL1(inputSample)
        if self.useBatchNorm:
            val = self.batchNormL1(val)
        val = torchFunc.prelu(val, self.preluL1.weight)

        # L2
        val = self.DenseL2(val)
        val = self.dropoutL2(val)
        if self.useBatchNorm:
            val = self.batchNormL2(val)
        val = torchFunc.prelu(val, self.preluL2.weight)

        # L3 
        #val = self.batchNormL3(val)
        val = self.DenseL3(val)
        val = self.dropoutL3(val)
        if self.useBatchNorm:
            val = self.batchNormL3(val)
        val = torchFunc.prelu(val, self.preluL3.weight)

        # L4
        val = self.DenseL4(val)
        val = self.dropoutL4(val)
        if self.useBatchNorm:
            val = self.batchNormL4(val)
        val = torchFunc.prelu(val, self.preluL4.weight)

        # L5
        val = self.DenseL5(val)
        val = self.dropoutL5(val)
        if self.useBatchNorm:
            val = self.batchNormL5(val)
        val = torchFunc.prelu(val, self.preluL5.weight)

        # L6
        val = self.DenseL6(val)
        val = self.dropoutL6(val)
        if self.useBatchNorm:
            val = self.batchNormL6(val)
        val = torchFunc.prelu(val, self.preluL6.weight)

        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Horizon Extraction Enhanced Fully Connected only, with Conic parameters from initial guess. EXPERIMENTAL: fully parametric model - 09-07-2024
class HorizonExtractionEnhancer_deepNNv8_fullyParametric(nn.Module):
    '''Horizon Extraction Enhanced Fully Connected only, with Conic parameters from initial guess. Parametric ReLU activations for all layers.
    Dropout and batch normalization for all layers. Fully parametric model with variable number of layers and sizes.'''
    def __init__(self, parametersConfig) -> None:
        super().__init__()

        # Extract all the inputs of the class init method from dictionary parametersConfig, else use default values
        useBatchNorm = parametersConfig.get('useBatchNorm', False)
        alphaDropCoeff = parametersConfig.get('alphaDropCoeff', 0)
        self.LinearInputSize = parametersConfig.get('LinearInputSize', 58)

        # Model parameters
        self.outChannelsSizes = parametersConfig.get('outChannelsSizes')
        self.num_layers = len(self.outChannelsSizes)

        self.useBatchNorm = useBatchNorm

        # Model architecture
        self.layers = nn.ModuleList()
        input_size = self.LinearInputSize # Initialize input size for first layer

        for i in range(self.num_layers):
            # Fully Connected layers block
            self.layers.append(nn.Linear(input_size, self.outChannelsSizes[i], bias=True))
            self.layers.append(nn.PReLU(self.outChannelsSizes[i]))
            self.layers.append(nn.Dropout(alphaDropCoeff))

            # Add batch normalization layer if required
            if self.useBatchNorm:
                self.layers.append(nn.BatchNorm1d(self.outChannelsSizes[i], eps=1E-5, momentum=0.1, affine=True))

            input_size = self.outChannelsSizes[i] # Update input size for next layer

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[-1], 2, bias=True)

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''

        for layer in self.layers:
            # Check if layer is a Linear layer
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu') # Apply Kaiming initialization
                if layer.bias is not None:
                    init.constant_(layer.bias, 0) # Initialize bias to zero if present


    def forward(self, inputSample):
        '''Forward pass method'''
        val = inputSample

        # Perform forward pass iterating through all layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                val = layer(val)
            elif isinstance(layer, nn.PReLU):
                val = torchFunc.prelu(val, layer.weight)
            elif isinstance(layer, nn.Dropout):
                val = layer(val)
            elif isinstance(layer, nn.BatchNorm1d):
                val = layer(val)

        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Horizon Extraction Enhancer CNNv7
class HorizonExtractionEnhancer_CNNv7(nn.Module):
    ''' '''
    def __init__(self, parametersConfig) -> None:
        super().__init__()

        # Extract all the inputs of the class init method from dictionary parametersConfig, else use default values

        kernelSizes = parametersConfig.get('kernelSizes', [1, 3, 3])
        poolkernelSizes = parametersConfig.get('poolkernelSizes', [1, 1, 2])

        useBatchNorm = parametersConfig.get('useBatchNorm', True)
        alphaDropCoeff = parametersConfig.get('alphaDropCoeff', 0)
        alphaLeaky = parametersConfig.get('alphaLeaky', 0.01)
        patchSize = parametersConfig.get('patchSize', 7)

        outChannelsSizes = parametersConfig.get('outChannelsSizes', [])

        #assert('LinearInputSkipSize' in parametersConfig.keys())
        if len(kernelSizes) != len(poolkernelSizes):
            raise ValueError('Kernel and pooling kernel sizes must have the same length')

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = len(kernelSizes)
        self.useBatchNorm = useBatchNorm

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolkernelSizes)

        #self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        self.LinearInputFeaturesSize = convBlockOutputSize[1] # convBlockOutputSize is tuple ((imgWidth, imgHeight), flattenedSize*nOutFeatures)
    
        self.LinearInputSkipSize = parametersConfig.get('LinearInputSkipSize') #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize
        


        #self.alphaLeaky = alphaLeaky

        # Model architecture
        idLayer = 0
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[idLayer], kernelSizes[0]) 
        self.preluL1 = nn.PReLU(self.outChannelsSizes[idLayer])
        self.maxPool2dL1 = nn.MaxPool2d(poolkernelSizes[idLayer], 1)
        idLayer += 1

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], kernelSizes[1]) 
        self.preluL2 = nn.PReLU(self.outChannelsSizes[idLayer])
        self.maxPool2dL2 = nn.MaxPool2d(poolkernelSizes[idLayer], 1)
        idLayer += 1

        self.conv2dL3 = nn.Conv2d(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], kernelSizes[2]) 
        self.preluL3 = nn.PReLU(self.outChannelsSizes[idLayer])
        self.maxPool2dL3 = nn.MaxPool2d(poolkernelSizes[idLayer], 1)
        idLayer += 1

        # Fully Connected predictor
        self.FlattenL3 = nn.Flatten()

        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[idLayer], bias=False)
        self.batchNormL4 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL4 = nn.PReLU()

        idLayer += 1

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], bias=True)
        self.batchNormL5 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL5 = nn.PReLU()

        idLayer += 1

        self.dropoutL6 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL6 = nn.Linear(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], bias=True)
        self.batchNormL6 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL6 = nn.PReLU()

        idLayer += 1

        self.dropoutL7 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL7 = nn.Linear(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], bias=True)
        self.batchNormL7 = nn.BatchNorm1d(self.outChannelsSizes[idLayer], eps=1E-5, momentum=0.1, affine=True)
        self.preluL7 = nn.PReLU()

        idLayer += 1

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[idLayer-1], 2, bias=True)

        # Initialize weights of layers
        #self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''
        # ReLU activation layers
        init.kaiming_uniform_(self.conv2dL1.weight, nonlinearity='leaky_relu') 
        init.constant_(self.conv2dL1.bias, 0)

        init.kaiming_uniform_(self.conv2dL2.weight, nonlinearity='leaky_relu') 
        init.constant_(self.conv2dL2.bias, 0)

        init.kaiming_uniform_(self.conv2dL3.weight, nonlinearity='leaky_relu')
        init.constant_(self.conv2dL3.bias, 0)

        init.kaiming_uniform_(self.DenseL4.weight, nonlinearity='leaky_relu')
        
        init.kaiming_uniform_(self.DenseL5.weight, nonlinearity='leaky_relu')
        init.constant_(self.DenseL5.bias, 0)

        init.kaiming_uniform_(self.DenseL6.weight, nonlinearity='leaky_relu')
        init.constant_(self.DenseL6.bias, 0)

        init.kaiming_uniform_(self.DenseL7.weight, nonlinearity='leaky_relu')
        init.constant_(self.DenseL7.bias, 0)

    def forward(self, inputSample):
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        
        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # Convolutional layers
        # L1 (Input)
        val = self.maxPool2dL1(torchFunc.prelu(self.conv2dL1(img2Dinput), self.preluL1.weight))

        # Fully Connected Layers
        # L2
        val = self.maxPool2dL2(torchFunc.prelu(self.conv2dL2(val), self.preluL2.weight))

        # L3
        val = self.maxPool2dL3(torchFunc.prelu( self.conv2dL3(val), self.preluL3.weight))

        # Fully Connected Layers
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4
        val = self.DenseL4(val)
        if self.useBatchNorm:
            val = self.batchNormL4(val)
        val = torchFunc.prelu(val, self.preluL4.weight)

        # L5
        val = self.DenseL5(val)
        val = self.dropoutL5(val)
        if self.useBatchNorm:
            val = self.batchNormL5(val)
        val = torchFunc.prelu(val, self.preluL5.weight)

        # L6
        val = self.DenseL6(val)
        val = self.dropoutL6(val)
        if self.useBatchNorm:
            val = self.batchNormL6(val)
        val = torchFunc.prelu(val, self.preluL6.weight)

        # L7
        val = self.DenseL7(val)
        val = self.dropoutL7(val)
        if self.useBatchNorm:
            val = self.batchNormL7(val)
        val = torchFunc.prelu(val, self.preluL7.weight)

        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Horizon Extraction Enhanced Fully Connected only, with Conic parameters from initial guess. EXPERIMENTAL: fully parametric model - 09-07-2024
class HorizonExtractionEnhancer_deepNNv8_fullyParametricNoImg(nn.Module):
    '''Horizon Extraction Enhanced Fully Connected only, with Conic parameters from initial guess WITHOUT image. Parametric ReLU activations for all layers.
    Dropout and batch normalization for all layers. Fully parametric model with variable number of layers and sizes.'''
    def __init__(self, parametersConfig) -> None:
        super().__init__()

        # Extract all the inputs of the class init method from dictionary parametersConfig, else use default values
        useBatchNorm = parametersConfig.get('useBatchNorm', False)
        alphaDropCoeff = parametersConfig.get('alphaDropCoeff', 0.1)
        self.LinearInputSize = parametersConfig.get('LinearInputSize', 58)

        # Model parameters
        self.outChannelsSizes = parametersConfig.get('outChannelsSizes')
        self.num_layers = len(self.outChannelsSizes)

        self.useBatchNorm = useBatchNorm

        # Model architecture
        self.layers = nn.ModuleList()
        input_size = self.LinearInputSize # Initialize input size for first layer

        for i in range(self.num_layers):
            # Fully Connected layers block
            self.layers.append(nn.Linear(input_size, self.outChannelsSizes[i], bias=True))
            self.layers.append(nn.PReLU(self.outChannelsSizes[i]))
            self.layers.append(nn.Dropout(alphaDropCoeff))

            # Add batch normalization layer if required
            if self.useBatchNorm:
                self.layers.append(nn.BatchNorm1d(self.outChannelsSizes[i], eps=1E-5, momentum=0.1, affine=True))

            input_size = self.outChannelsSizes[i] # Update input size for next layer

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[-1], 2, bias=True)

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''

        for layer in self.layers:
            # Check if layer is a Linear layer
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu') # Apply Kaiming initialization
                if layer.bias is not None:
                    init.constant_(layer.bias, 0) # Initialize bias to zero if present


    def forward(self, inputSample):
        '''Forward pass method'''
        val = inputSample

        # Perform forward pass iterating through all layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                val = layer(val)
            elif isinstance(layer, nn.PReLU):
                val = torchFunc.prelu(val, layer.weight)
            elif isinstance(layer, nn.Dropout):
                val = layer(val)
            elif isinstance(layer, nn.BatchNorm1d):
                val = layer(val)

        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Horizon Extraction Enhancer CNNvX_fullyParametric - 16-09-2024
class HorizonExtractionEnhancer_CNNvX_fullyParametric(nn.Module):
    ''' '''

    def __init__(self, parametersConfig) -> None:
        super().__init__()

        # Extract all the inputs of the class init method from dictionary parametersConfig, else use default values

        kernelSizes = parametersConfig.get('kernelSizes', [5, 3, 3])
        poolkernelSizes = parametersConfig.get('poolkernelSizes', [2, 2, 2])

        useBatchNorm = parametersConfig.get('useBatchNorm', True)
        alphaDropCoeff = parametersConfig.get('alphaDropCoeff', 0)
        alphaLeaky = parametersConfig.get('alphaLeaky', 0)
        patchSize = parametersConfig.get('patchSize', 7)

        outChannelsSizes = parametersConfig.get('outChannelsSizes', [])

        if len(kernelSizes) != len(poolkernelSizes):
            raise ValueError('Kernel and pooling kernel sizes must have the same length')

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = len(kernelSizes)
        self.useBatchNorm = useBatchNorm

        self.num_layers = len(self.outChannelsSizes) - len(kernelSizes)

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolkernelSizes)

        # self.LinearInputFeaturesSize = (patchSize - self.numOfConvLayers * np.floor(float(kernelSizes[-1])/2.0)) * self.outChannelsSizes[-1] # Number of features arriving as input to FC layer
        # convBlockOutputSize is tuple ((imgWidth, imgHeight), flattenedSize*nOutFeatures)
        self.LinearInputFeaturesSize = convBlockOutputSize[1]

        # 11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSkipSize = parametersConfig.get('LinearInputSkipSize')
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.layers = nn.ModuleList()
        input_size = self.LinearInputSize  # Initialize input size for first layer

        # Model architecture
        idLayer = 0

        # Convolutional Features extractor
        # Conv block 1
        #self.layers.append(nn.Conv2d(1, self.outChannelsSizes[idLayer], kernelSizes[0]))
        #self.layers.append(nn.PReLU(self.outChannelsSizes[idLayer]))
        #idLayer += 1

        # Conv block 2
        #self.layers.append(nn.Conv2d(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], kernelSizes[1]))
        #self.layers.append(nn.PReLU(self.outChannelsSizes[idLayer]))
        #idLayer += 1

        # Conv block 3
        #self.layers.append(nn.Conv2d(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], kernelSizes[2]))
        #self.layers.append(nn.PReLU(self.outChannelsSizes[idLayer]))
        #idLayer += 1

        # Convolutional Features extractor
        #self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[idLayer], kernelSizes[0]) 
        #self.preluL1 = nn.PReLU(self.outChannelsSizes[idLayer])
        #self.maxPool2dL1 = nn.MaxPool2d(
        #    poolkernelSizes[idLayer], poolkernelSizes[idLayer])
        #idLayer += 1
        #self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], kernelSizes[1]) 
        #self.preluL2 = nn.PReLU(self.outChannelsSizes[idLayer])
        #self.maxPool2dL2 = nn.MaxPool2d(
        #    poolkernelSizes[idLayer], poolkernelSizes[idLayer])
        #idLayer += 1
        #self.conv2dL3 = nn.Conv2d(self.outChannelsSizes[idLayer-1], self.outChannelsSizes[idLayer], kernelSizes[2]) 
        #self.preluL3 = nn.PReLU(self.outChannelsSizes[idLayer])
        #self.maxPool2dL3 = nn.MaxPool2d(
        #    poolkernelSizes[idLayer], poolkernelSizes[idLayer])

        # Convolutional blocks autobuilder
        in_channels = 1

        for i in range(len(kernelSizes)):
            # Convolutional layers block
            self.layers.append( nn.Conv2d(in_channels, self.outChannelsSizes[i], kernelSizes[i]))
            self.layers.append(nn.PReLU(self.outChannelsSizes[i]))
            self.layers.append(nn.MaxPool2d( poolkernelSizes[i], poolkernelSizes[i]))

            in_channels = self.outChannelsSizes[i]
            idLayer += 1

        # Fully Connected predictor autobuilder
        #self.Flatten = nn.Flatten()
        self.layers.append(nn.Flatten())

        input_size = self.LinearInputSize  # Initialize input size for first layer

        for i in range(idLayer, self.num_layers+idLayer):
            # Fully Connected layers block
            self.layers.append(nn.Linear(input_size, self.outChannelsSizes[i], bias=True))
            self.layers.append(nn.PReLU(self.outChannelsSizes[i]))
            self.layers.append(nn.Dropout(alphaDropCoeff))

            # Add batch normalization layer if required
            if self.useBatchNorm:
                self.layers.append(nn.BatchNorm1d(self.outChannelsSizes[i], eps=1E-5, momentum=0.1, affine=True))

            # Update input size for next layer
            input_size = self.outChannelsSizes[i]

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[-1], 2, bias=True)

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''
        # ReLU activation layers
        #init.kaiming_uniform_(self.conv2dL1.weight, nonlinearity='leaky_relu')
        #init.constant_(self.conv2dL1.bias, 0)
        #init.kaiming_uniform_(self.conv2dL2.weight, nonlinearity='leaky_relu')
        #init.constant_(self.conv2dL2.bias, 0)
        #init.kaiming_uniform_(self.conv2dL3.weight, nonlinearity='leaky_relu')
        #init.constant_(self.conv2dL3.bias, 0)

        for layer in self.layers:
            # Check if layer is a Linear layer
            if isinstance(layer, nn.Linear):
                # Apply Kaiming initialization
                init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    # Initialize bias to zero if present
                    init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.Conv2d):
                # Apply Kaiming initialization
                init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    # Initialize bias to zero if present
                    init.constant_(layer.bias, 0)

    def forward(self, inputSample):

        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        # img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D

        # assert (inputSample.size(1) == ( self.imagePixSize + self.LinearInputSkipSize))
        # img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D

        imgWidth = int(sqrt(self.imagePixSize))
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape( imgWidth, -1, 1, inputSample.size(0))).T  # First portion of the input vector reshaped to 2D
    
        # Step 1: Select the first self.imagePixSize columns for all rows
        # Step 2: Permute the dimensions to match the transposition (swap axes 0 and 1)
        # Step 3: Reshape the permuted tensor to the specified dimensions
        # Step 4: Permute again to match the final transposition (swap axes 0 and 1 again)

        img2Dinput = (((inputSample[:, 0:self.imagePixSize]).permute(1, 0)).reshape(
            imgWidth, -1, 1, inputSample.size(0))).permute(3, 2, 1, 0)

        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # Perform forward pass iterating through all layers of CNN
        val = img2Dinput

        for layer in self.layers:

            if isinstance(layer, nn.Conv2d):
                val = layer(val)
            elif isinstance(layer, nn.MaxPool2d):
                val = layer(val)
            elif isinstance(layer, nn.Linear):
                val = layer(val)
            elif isinstance(layer, nn.PReLU):
                val = torchFunc.prelu(val, layer.weight)
            elif isinstance(layer, nn.Dropout):
                val = layer(val)
            elif isinstance(layer, nn.BatchNorm1d):
                val = layer(val)
            elif isinstance(layer, nn.Flatten):
                val = layer(val)
                # Concatenate if needed
                if 'contextualInfoInput' in locals():
                    val = torch.cat((val, contextualInfoInput), dim=1)


        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
