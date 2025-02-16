# Script created by PeterC 21-05-2024 to experiment with Limb-based navigation using CNN-NN network
# Reference forum discussion: https://discuss.pytorch.org/t/adding-input-layer-after-a-hidden-layer/29225
# Prototype architecture designed and coded, 03-05-2024

# BRIEF DESCRIPTION
# The network takes windows of Moon images where the illuminated limb is present, which may be a feature map already identifying the limb if edge detection has been performed.
# Convolutional layers process the image patch to extract features, then stacked with several other inputs downstream. Among these, the relative attitude of the camera wrt to the target body 
# and the Sun direction in the image plane. A series of fully connected layers then map the flatten 1D vector of features plus the contextual information to infer a correction of the edge pixels.
# This correction is intended to remove the effect of the imaged body morphology (i.e., the network will account for the DEM "backward") such that the refined edge pixels better adhere to the
# assumption of ellipsoidal/spherical body without terrain effects. The extracted pixels are then used by the Christian Robinson algorithm to perform derivation of the position vector.

# To insert multiple inputs at different layers of the network, following the discussion:
# 1) Construct the input tensor X such that it can be sliced into the image and the other data
# 2) Define the architecture class
# 3) In defining the forward(self, x) function, slice the x input according to the desired process, reshape and do other operations. 
# 4) Concatenate the removed data with the new inputs from the preceding layers as input to where they need to go.

# Import modules
import os
import matplotlib.pyplot as plt
import torch, os
import pyTorchAutoForge
from pyTorchAutoForge.modelBuilding.ModelAutoBuilder import AutoComputeConvBlocksOutput

from torch import nn
from math import sqrt

from typing import Union
import numpy as np

import torch.nn.functional as torchFunc # Module to apply activation functions in forward pass instead of defining them in the model class

# Possible option: X given as input to the model is just one, but managed internally to the class, thus splitting the input as appropriate and only used in the desired layers.
# Alternatively, forward() method can accept multiple inputs based on the definition.

# DEVNOTE: check which modifications are needed for training in mini-batches
# According to GPT4o, the changes required fro the training in batches are not many. Simply build the dataset accordingly, so that Torch is aware of the split:
# Exameple: the forward() method takes two input vectors: forward(self, main_inputs, additional_inputs)
# main_inputs = torch.randn(1000, 784)  # Main input (e.g., image features)
# additional_inputs = torch.randn(1000, 10)  # Additional input (e.g., metadata)
# labels = torch.randint(0, 10, (1000,))
# dataset = TensorDataset(main_inputs, additional_inputs, labels)

# %% Auxiliary functions

def create_custom_mosaic_with_points_gray(image_array, positions, large_image_size, patch_points, global_points=None, save_path=None):
    # Get dimensions from the input array
    img_height, img_width, num_images = image_array.shape

    # Initialize the large image with a fixed size
    large_image = np.zeros(large_image_size, dtype=np.uint8)

    # Place each image in the large image at specified positions
    for idx in range(num_images):
        center_y, center_x = positions[idx]

        # Calculate the top-left corner from the center position
        start_y = int(center_y - np.floor(img_height / 2))
        start_x = int(center_x - np.floor(img_width / 2))

        # Ensure the patch fits within the large image dimensions
        if start_y >= 0 and start_x >= 0 and start_y + img_height <= large_image_size[0] and start_x + img_width <= large_image_size[1]:
            # Debug output
            print(f'Placing image {idx+1} at position ({start_y}, {start_x})')

            # Place the patch
            large_image[start_y:start_y+img_height,
                        start_x:start_x+img_width] = image_array[:, :, idx]
        else:
            print(
                f'Warning: Image {idx+1} at position ({start_y}, {start_x}) exceeds the large image bounds and will not be placed.')

    # Display the large image with patches
    plt.figure()
    plt.imshow(large_image, cmap='gray')
    plt.title('Custom Mosaic Image with Points')

    # Plot the points corresponding to each patch
    for idx in range(num_images):
        point_y, point_x = patch_points[idx]
        plt.plot(point_x, point_y, 'ro', markersize=5, linewidth=0.5)

    # Plot the additional global points if provided
    if global_points is not None:
        for idx in range(len(global_points)):
            point_y, point_x = global_points[idx]
            plt.plot(point_x, point_y, 'bo', markersize=10, linewidth=0.5)

    # Save the figure if save_path is provided
    if save_path is not None:
        # Ensure the save directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save the image
        save_file = os.path.join(save_path, 'mosaic.png')
        plt.savefig(save_file, bbox_inches='tight')
        print(f'Mosaic saved to {save_file}')

    plt.show()

# Example usage
#if __name__ == "__main__":
#    # Example: create a random array of images for demonstration
#    height = 100
#    width = 100
#    num_images = 5
#    image_array = np.random.randint(
#        0, 256, (height, width, num_images), dtype=np.uint8)
#
#    # Example positions (top-left corners) for each patch
#    positions = np.array([[1, 1], [1, 150], [150, 1], [150, 150], [75, 75]])
#
#    # Example points corresponding to each patch
#    patch_points = np.array(
#        [[50, 50], [50, 200], [200, 50], [200, 200], [125, 125]])
#
#    # Example additional global points
#    global_points = np.array([[30, 30], [70, 250], [250, 70], [270, 270]])
#
#    # Size of the large image
#    large_image_size = (300, 300)
#
#    # Path to save the mosaic
#    save_path = 'mosaic_images'
#
#    create_custom_mosaic_with_points_gray(
#        image_array, positions, large_image_size, patch_points, global_points, save_path)



#######################################################################################################
# %% Function to validate path (check it is not completely black or white)
def IsPatchValid(patchFlatten, lowerIntensityThr=3):
    
    # Count how many pixels are below threshold
    howManyBelowThreshold = np.sum(patchFlatten <= lowerIntensityThr)
    howManyPixels = len(patchFlatten)
    width = np.sqrt(howManyPixels)
    lowerThreshold = width/2
    upperThreshold = howManyPixels - lowerThreshold
    if howManyBelowThreshold <  lowerThreshold or howManyBelowThreshold > upperThreshold:
        return False
    else:
        return True

# %% ARCHITECTURES ############################################################################################################


# %% Custom CNN-NN model for Moon limb pixel extraction enhancer - 01-06-2024
'''Architecture characteristics: Conv. layers, average pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization.
Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates.
'''
class HorizonExtractionEnhancerCNNv1avg(nn.Module):
    def __init__(self, outChannelsSizes:list, kernelSizes, poolingKernelSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2

        convBlockOutputSize = AutoComputeConvBlocksOutput(
                                self, kernelSizes, poolingKernelSize)
        self.LinearInputFeaturesSize = convBlockOutputSize[1]

        self.LinearInputSkipSize = 7 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.avgPoolL1 = nn.AvgPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.avgPoolL2 = nn.AvgPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))

        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D

        contextualInfoInput = inputSample[:, self.imagePixSize:]


        # Convolutional layers
        # L1 (Input)
        val = self.avgPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))
        # L2
        val = self.avgPoolL2(torchFunc.leaky_relu(self.conv2dL2(val), self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Custom training function for Moon limb pixel extraction enhancer V2 (with target average radius in pixels) - 21-06-2024
class HorizonExtractionEnhancerCNNv2avg(nn.Module):
    '''
    Architecture characteristics: Conv. layers, average pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates, target average radius in pixels.
    '''
    def __init__(self, outChannelsSizes:list, kernelSizes, poolingKernelSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)
        self.LinearInputFeaturesSize = convBlockOutputSize[1]
        
        self.LinearInputSkipSize = 8 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.avgPoolL1 = nn.AvgPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.avgPoolL2 = nn.AvgPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        
        
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))

        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D

        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # Convolutional layers
        # L1 (Input)
        val = self.avgPoolL1(torchFunc.leaky_relu(self.conv2dL1(img2Dinput), self.alphaLeaky))
        # L2
        val = self.avgPoolL2(torchFunc.leaky_relu(self.conv2dL2(val), self.alphaLeaky))

        # Fully Connected Layers
        # L3
        val = self.FlattenL3(val) # Flatten data to get input to Fully Connected layers

        # Concatenate and batch normalize data
        val = torch.cat((val, contextualInfoInput), dim=1)

        # L4 
        #val = self.batchNormL3(val)
        val = self.dropoutL4(val)
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    
# %% Custom CNN-NN model for Moon limb pixel extraction enhancer V1max - 23-06-2024
    '''Architecture characteristics: Conv. layers, max pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization.
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates.
    '''
class HorizonExtractionEnhancerCNNv1max(nn.Module):

    def __init__(self, outChannelsSizes:list, kernelSizes, poolingKernelSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2

        convBlockOutputSize = AutoComputeConvBlocksOutput(self, kernelSizes, poolingKernelSize)
        self.LinearInputFeaturesSize = convBlockOutputSize[1]

        self.LinearInputSkipSize = 7 #11 # CHANGE TO 7 removing R_DEM and PosTF
        self.LinearInputSize = self.LinearInputSkipSize + self.LinearInputFeaturesSize

        self.alphaLeaky = alphaLeaky

        # Model architecture
        # Convolutional Features extractor
        self.conv2dL1 = nn.Conv2d(1, self.outChannelsSizes[0], kernelSizes[0]) 
        self.maxPoolL1 = nn.MaxPool2d(poolingKernelSize, 1)

        self.conv2dL2 = nn.Conv2d(self.outChannelsSizes[0], self.outChannelsSizes[1], kernelSizes[1]) 
        self.maxPoolL2 = nn.MaxPool2d(poolingKernelSize, 1) 

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

    def forward(self, inputSample):
        # Extract image and contextual information from inputSample
        # ACHTUNG: transpose, reshape, transpose operation assumes that input vector was reshaped column-wise (FORTRAN style)
        #img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape(int(np.sqrt(float(self.imagePixSize))), -1, 1, inputSample.size(0))).T # First portion of the input vector reshaped to 2D
        
        assert(inputSample.size(1) == (self.imagePixSize + self.LinearInputSkipSize))

        #img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(int(torch.sqrt( torch.tensor(self.imagePixSize) )), -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        imgWidth = int(sqrt( self.imagePixSize ))
        img2Dinput =  ( ( (inputSample[:, 0:self.imagePixSize]).T).reshape(imgWidth, -1, 1, inputSample.size(0) ) ).T # First portion of the input vector reshaped to 2D
        contextualInfoInput = inputSample[:, self.imagePixSize:]

        # DEBUG
        #print(img2Dinput[0, 0, :,:])
        ########################################

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
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    

# %% Custom training function for Moon limb pixel extraction enhancer V2max (with target average radius in pixels) - 23-06-2024
'''
Architecture characteristics: Conv. layers, max pooling, fully connected layers, dropout, leaky ReLU activation, batch normalization
    Input: Image patch with Moon limb, contextual information: relative attitude, sun direction in pixels, patch centre coordinates, target average radius in pixels.
'''
class HorizonExtractionEnhancerCNNv2max(nn.Module):
    def __init__(self, outChannelsSizes:list, kernelSizes, poolingKernelSize=2, alphaDropCoeff=0.1, alphaLeaky=0.01, patchSize=7) -> None:
        super().__init__()

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = 2

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

        # Fully Connected predictor
        # NOTE: Add batch normalization here
        self.FlattenL3 = nn.Flatten()
        #self.batchNormL3 = nn.BatchNorm1d(int(self.LinearInputSize), eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable
        #self.batchNormL3 = nn.BatchNorm1d(41, eps=1E-5, momentum=0.1, affine=True) # affine=True set gamma and beta parameters as learnable

        self.dropoutL4 = nn.Dropout2d(alphaDropCoeff)
        self.DenseL4 = nn.Linear(int(self.LinearInputSize), self.outChannelsSizes[2], bias=True)

        self.dropoutL5 = nn.Dropout1d(alphaDropCoeff)
        self.DenseL5 = nn.Linear(self.outChannelsSizes[2], self.outChannelsSizes[3], bias=True)

        # Output layer
        self.DenseOutput = nn.Linear(self.outChannelsSizes[3], 2, bias=True)

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
        val = torchFunc.leaky_relu(self.DenseL4(val), self.alphaLeaky)
        # L5
        val = self.dropoutL5(val)
        val = torchFunc.leaky_relu(self.DenseL5(val), self.alphaLeaky)
        # Output layer
        predictedPixCorrection = self.DenseOutput(val)

        return predictedPixCorrection
    
