# Module to apply activation functions in forward pass instead of defining them in the model class
from typing import Union
from pyTorchAutoForge.api.torch import * 
from pyTorchAutoForge.modelBuilding.ModelAutoBuilder import AutoComputeConvBlocksOutput, ComputeConv2dOutputSize, ComputePooling2dOutputSize, ComputeConvBlockOutputSize
from pyTorchAutoForge.api.torch.torchModulesIO import SaveTorchModel, LoadTorchModel

from torch import nn
from torch.nn import init
from torch.nn import functional as torchFunc


# DEVNOTE TODO change name of this file to "modelBuildingBlocks.py" and move the OLD classes to the file "modelClasses.py" for compatibility with legacy codebase
 
#############################################################################################################################################
class torchModel(torch.nn.Module):
    '''Custom base class inheriting nn.Module to define a PyTorch NN model, augmented with saving/loading routines like Pytorch Lightning.'''

    def __init__(self, moduleName:str = None, enable_tracing:bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Assign module name. If not provided by user, use class name
        if moduleName is None:
            self.moduleName = self.__class__.__name__
        else:
            self.moduleName  = moduleName


    def saveState(self, exampleInput = None, target_device:str = None) -> None:

        if self.enable_tracing == True and exampleInput is None:
            self.enable_tracing = False
            raise Warning('You must provide an example input to trace the model through torch.jit.trace(). Overriding enable_tracing to False.')
        
        if target_device is None:
            target_device = self.device

        SaveTorchModel(modelName = self.moduleName, 
                        saveAsTraced = self.enable_tracing,
                        exampleInput = exampleInput,
                       targetDevice = target_device)

    def loadState(self):
        
        LoadTorchModel()
#############################################################################################################################################
# TBC: class to perform code generation of net classes instead of classes with for and if loops? 
# --> THe key problem with the latter is that tracing/scripting is likely to fail due to conditional statements

# TODO The structure of the model building blocks should be as follows:
# Normalization Layer example:

# TBC: classes versus functions?

class NormalizationLayer(): # DEVNOTE: how to pass arguments?
    def __init__(self, dict_key, *args, **kwargs) -> nn.Module:

        self.modules_map = nn.ModuleDict(
                                ['BatchNorm2d', nn.BatchNorm2d],
                                ['LayerNorm', nn.LayerNorm],
                                ['InstanceNorm2d', nn.InstanceNorm2d],
                                ['GroupNorm', nn.GroupNorm]
                                )
        
        return self.modules_map[dict_key](args) # TBC this is ok?

# TODO
class ActivationLayer():
    def __init__(self, dict_key, *args, **kwargs) -> nn.Module:
        self.modules_map = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU],
            ['relu', nn.ReLU],
            ['prelu', nn.PReLU],
        ])

        return self.modules_map[dict_key](args)

# TODO 
class ConvolutionalLayer():
    def __init__(self, dict_key, *args, **kwargs) -> nn.Module:
        pass

# TODO --> convolutional building block
class ConvolutionalBlock():
    def __init__(self, dict_key, *args, **kwargs) -> nn.Sequential:
        

        return nn.Sequential() 



# %% TemplateConvNet - 19-09-2024
class TemplateConvNet(torchModel):
    '''Template class for a fully parametric CNN model in PyTorch. Inherits from torchModel class (nn.Module enhanced class).'''
    # TODO: not finished yet
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
            raise ValueError(
                'Kernel and pooling kernel sizes must have the same length')

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.patchSize = patchSize
        self.imagePixSize = self.patchSize**2
        self.numOfConvLayers = len(kernelSizes)
        self.useBatchNorm = useBatchNorm

        self.num_layers = len(self.outChannelsSizes) - len(kernelSizes)

        convBlockOutputSize = AutoComputeConvBlocksOutput(
            self, kernelSizes, poolkernelSizes)

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

        # Convolutional blocks auto building
        in_channels = 1

        for i in range(len(kernelSizes)):
            # Convolutional layers block
            self.layers.append(
                nn.Conv2d(in_channels, self.outChannelsSizes[i], kernelSizes[i]))
            self.layers.append(nn.PReLU(self.outChannelsSizes[i]))
            self.layers.append(nn.MaxPool2d(
                poolkernelSizes[i], poolkernelSizes[i]))

            in_channels = self.outChannelsSizes[i]
            idLayer += 1

        # Fully Connected predictor autobuilder
        # self.Flatten = nn.Flatten()
        self.layers.append(nn.Flatten())

        input_size = self.LinearInputSize  # Initialize input size for first layer

        for i in range(idLayer, self.num_layers+idLayer):
            # Fully Connected layers block
            self.layers.append(
                nn.Linear(input_size, self.outChannelsSizes[i], bias=True))
            self.layers.append(nn.PReLU(self.outChannelsSizes[i]))
            self.layers.append(nn.Dropout(alphaDropCoeff))

            # Add batch normalization layer if required
            if self.useBatchNorm:
                self.layers.append(nn.BatchNorm1d(
                    self.outChannelsSizes[i], eps=1E-5, momentum=0.1, affine=True))

            # Update input size for next layer
            input_size = self.outChannelsSizes[i]

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''

         # Wait, why is it using onlt Kaiming?
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

        imgWidth = int(torch.sqrt(self.imagePixSize))
        # img2Dinput = (((inputSample[:, 0:self.imagePixSize]).T).reshape( imgWidth, -1, 1, inputSample.size(0))).T  # First portion of the input vector reshaped to 2D

        # Step 1: Select the first self.imagePixSize columns for all rows
        # Step 2: Permute the dimensions to match the transposition (swap axes 0 and 1)
        # Step 3: Reshape the permuted tensor to the specified dimensions
        # Step 4: Permute again to match the final transposition (swap axes 0 and 1 again)

        # Perform forward pass iterating through all layers of CNN
        val = inputSample

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

        # Output layer
        predictedPixCorrection = val

        return predictedPixCorrection


# %% TemplateDeepNet - 19-09-2024
class TemplateDeepNet(torchModel):
    '''Template class for a fully parametric Deep NN model in PyTorch. Inherits from torchModel class (nn.Module enhanced class).'''

    def __init__(self, parametersConfig) -> None:
        super().__init__()

        useBatchNorm = parametersConfig.get('useBatchNorm', True)
        alphaDropCoeff = parametersConfig.get('alphaDropCoeff', 0)
        alphaLeaky = parametersConfig.get('alphaLeaky', 0)
        outChannelsSizes = parametersConfig.get('outChannelsSizes', [])
        
        # Initialize input size for first layer
        input_size = parametersConfig.get('input_size')

        # Model parameters
        self.outChannelsSizes = outChannelsSizes
        self.useBatchNorm = useBatchNorm

        self.num_layers = len(self.outChannelsSizes)

        # Model architecture
        self.layers = nn.ModuleList()
        idLayer = 0

        # Fully Connected autobuilder
        self.layers.append(nn.Flatten())


        for i in range(idLayer, self.num_layers+idLayer):

            # Fully Connected layers block
            self.layers.append(nn.Linear(input_size, self.outChannelsSizes[i], bias=True))
            self.layers.append(nn.PReLU(self.outChannelsSizes[i]))
            self.layers.append(nn.Dropout(alphaDropCoeff))

            # Add batch normalization layer if required
            if self.useBatchNorm:
                self.layers.append(nn.BatchNorm1d(
                    self.outChannelsSizes[i], eps=1E-5, momentum=0.1, affine=True))

            # Update input size for next layer
            input_size = self.outChannelsSizes[i]

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''

        for layer in self.layers:
            # Check if layer is a Linear layer
            if isinstance(layer, nn.Linear):
                # Apply Kaiming initialization
                init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    # Initialize bias to zero if present
                    init.constant_(layer.bias, 0)

    def forward(self, inputSample):
        # Perform forward pass iterating through all layers of DNN
        val = inputSample
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                val = layer(val)
            elif isinstance(layer, nn.PReLU):
                val = torchFunc.prelu(val, layer.weight)
            elif isinstance(layer, nn.Dropout):
                val = layer(val)
            elif isinstance(layer, nn.BatchNorm1d):
                val = layer(val)
            elif isinstance(layer, nn.Flatten):
                val = layer(val)

        # Output layer
        prediction = val

        return prediction
