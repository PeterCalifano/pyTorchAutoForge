# Module to apply activation functions in forward pass instead of defining them in the model class
import torch.nn as nn
from typing import Union
from pyTorchAutoForge.api.torch import * 
from pyTorchAutoForge.model_building.ModelAutoBuilder import AutoComputeConvBlocksOutput, ComputeConv2dOutputSize, ComputePooling2dOutputSize, ComputeConvBlockOutputSize, enumMultiHeadOutMode, MultiHeadRegressor
from pyTorchAutoForge.api.torch.torchModulesIO import SaveTorchModel, LoadTorchModel
from pyTorchAutoForge.model_building.modelBuildingFunctions import build_activation_layer
from pyTorchAutoForge.model_building.ModelMutator import ModelMutator
import inspect, pytest
from torch import nn
from torch.nn import init
from torch.nn import functional as torchFunc
import torch, optuna, os, kornia
import numpy as np
from torchvision import models

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

        SaveTorchModel(modelpath = self.moduleName, 
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


        for i in range(idLayer, self.num_layers+idLayer-1):

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

        # Add output layer
        self.layers.append(nn.Linear(input_size, self.outChannelsSizes[-1], bias=True))

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


# DEVELOPMENT CODE: DEVNOTE: test definition of template DNN using new build_activation_layer function
class TemplateDeepNet_experimental(torchModel):
    '''Template class for a fully parametric Deep NN model in PyTorch. Inherits from torchModel class (nn.Module enhanced class).'''

    def __init__(self, parametersConfig) -> None:
        super().__init__()

        useBatchNorm = parametersConfig.get('useBatchNorm', False) # TODO try to replace with build_normalization_layer function
        alphaDropCoeffLayers = parametersConfig.get('alphaDropCoeffLayers', None) # Can be either scalar (apply to all) or list (apply to specific layers)
        #alphaLeaky = parametersConfig.get('alphaLeaky', 0)
        outChannelsSizes = parametersConfig.get('outChannelsSizes', [])

        if alphaDropCoeffLayers is not None:
            assert len(alphaDropCoeffLayers) == len(outChannelsSizes) -1, 'Length of alphaDropCoeffLayers must match number of layers in outChannelsSizes'

        # Define activation function parameters (default: PReLU)
        self.activation_fcn_name = parametersConfig.get( 'activation_fcn_name', 'PReLU')
        act_fcn_params_dict = parametersConfig.get( 'act_fcn_params_dict', {'num_parameters': 'all'})

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

        for i in range(idLayer, self.num_layers+idLayer-1):

            # Build Linear layer
            self.layers.append( nn.Linear(input_size, self.outChannelsSizes[i], bias=True))

            # Build activation layer
            if self.activation_fcn_name == 'PReLU': 
                act_fcn_params_dict['num_parameters'] = self.outChannelsSizes[i]

            self.layers.append(build_activation_layer( self.activation_fcn_name, False, **act_fcn_params_dict))

            # Add dropout layer if required
            if alphaDropCoeffLayers is not None:
                if len(alphaDropCoeffLayers) > 0 and len(alphaDropCoeffLayers) == 1:
                    self.layers.append(nn.Dropout(alphaDropCoeffLayers[0])) # Add to all layers

                if alphaDropCoeffLayers[i] > 0:
                    self.layers.append(nn.Dropout(alphaDropCoeffLayers[i])) # Add to layer as specified by user

            # Add batch normalization layer if required
            if self.useBatchNorm:
                self.layers.append(nn.BatchNorm1d( self.outChannelsSizes[i], eps=1E-5, momentum=0.1, affine=True))

            # Update input size for next layer
            input_size = self.outChannelsSizes[i]

        # Add output layer
        self.layers.append(nn.Linear(input_size, self.outChannelsSizes[-1], bias=True))

        # Initialize weights of layers
        self.__initialize_weights__()

    def __initialize_weights__(self):
        '''Weights Initialization function for layers of the model. Xavier --> layers with tanh and sigmoid, Kaiming --> layers with ReLU activation'''

        for layer in self.layers:

            # Check if layer is a Linear layer
            if isinstance(layer, nn.Linear):
                # Apply Kaiming initialization
                if self.activation_fcn_name.lower() in ['relu', 'leakyrelu', 'prelu']:
                    init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
                elif self.activation_fcn_name.lower() in ['tanh', 'sigmoid']:
                    init.xavier_uniform_(layer.weight)

                if layer.bias is not None:
                    # Initialize bias to zero if present
                    init.constant_(layer.bias, 0)

    def forward(self, x):
        # Perform forward pass iterating through all layers of DNN
        for layer in self.layers:
            x = layer(x)
        return x 


class EfficientNetBackbone(nn.Module):
    def __init__(self, efficient_net_ID, output_type: str = 'last', device='cpu'):
        super(EfficientNetBackbone, self).__init__()

        self.output_type = output_type
        self.device = device
        self.features = []

        if efficient_net_ID == 0:
            self.modelType = models.efficientnet_b0
        elif efficient_net_ID == 1:
            self.modelType = models.efficientnet_b1

        # Remove last Linear classifier and dropout layer (Classifier nn.Sequential module)
        if self.output_type == 'last':
            self.feature_extractor = nn.ModuleList(nn.Sequential(
                *list((self.modelType(weights=True).to(self.device)).children())[:-1]))

        elif self.output_type == 'features':
            # self._register_hooks()
            self.feature_extractor = nn.ModuleList(
                list((self.modelType(weights=True).to(self.device)).children())[0].children())

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        for name, layer in self.feature_extractor.named_children():
            if 'blocks' in name:  # Register hook for each block layer
                for idx, block in enumerate(layer):
                    block.register_forward_hook(get_activation(f'block_{idx}'))

    def forward(self, x):
        self.features = []  # Reset features list state
        for module in self.feature_extractor:
            # Evaluate each module in the feature extractor
            x = module(x)
            # If output type is 'features', append to list to store intermediate features
            if self.output_type == 'features':
                self.features.append(x)

        return x if self.output_type == 'last' else self.features

# RESOLUTION ADAPTERS
class conv2dResolutionAdapter(nn.Module):
    """
    conv2dResolutionAdapter _summary_

    _extended_summary_

    :param nn: _description_
    :type nn: _type_
    """

    def __init__(self, targetDimsInPix: Union[list, np.ndarray, torch.Tensor],
                 channelInOutSizes: Union[list, np.ndarray, torch.Tensor] = [1, 3]):
        super().__init__()

        # Perform 1D convolution to get three feature maps
        self.channelExpander = torch.nn.Conv2d(
            channelInOutSizes[0], channelInOutSizes[1], kernel_size=1, stride=2, padding=0, bias=False)

        # Define adapter model to bring resolution down to feature_extractor input size
        self.adaptive_pool_L0 = torch.nn.AdaptiveAvgPool2d(
            output_size=(targetDimsInPix[0], targetDimsInPix[1]))

    def forward(self, inputImage):

        # Forward pass of the adapter model
        x = self.channelExpander(inputImage)
        x = self.adaptive_pool_L0(x)

        return x



class resizeCopyAdapter(nn.Module):
    """
    resizeCopyAdapter Torch module working as size and channels adapter for input images. It resizes input images to a target size (default: EfficientNet_B0 input size) and copies the data along the channel dimension if necessary.

    :param nn: _description_
    :type nn: _type_
    """
    def __init__(self, output_size: list = [224, 224], num_channels: list = [1, 3], interp_method: str = 'bilinear'):

        super(resizeCopyAdapter, self).__init__()
        self.output_size = output_size
        self.input_channels, self.output_channels = num_channels
        self.interp_method = interp_method

    def forward(self, x):
        # Resize to output size
        x = kornia.geometry.transform.resize(
            x, self.output_size, interpolation=self.interp_method)

        # Copy tensor data along channels size if necessary
        if self.output_channels > self.input_channels:
            x = x.repeat(1, self.output_channels // self.input_channels, 1, 1)

        return x

# %% TEMPORARY DEV (from OPERATIVE-DEVELOP)

class MultiScaleRangeRegressor(nn.Module):
    def __init__(self, headRange_config: dict):
        super(MultiScaleRangeRegressor, self).__init__()

        self.feature_merger = FeatureMerger_tailoredForV2()  # TODO
        self.headRange = TemplateDeepNet(headRange_config)

    def forward(self, Xfeatures: list):
        x = self.feature_merger(Xfeatures)
        return self.headRange(x)



class MultiHead_CentroidRange_V1(MultiHeadRegressor):
    # Centroid head: straight DNN from last layer of backbone
    # Range head:  straight DNN from last layer of backbone
    def __init__(self, model_heads: Union[nn.ModuleList, nn.ModuleDict, nn.Module, dict]):
        super().__init__(model_heads, enumMultiHeadOutMode.Concatenate)

    def forward(self, Xfeatures : Union[list, dict, torch.tensor]):
        if isinstance(Xfeatures, dict):
            raise NotImplementedError("Not implemented yet")
            X = Xfeatures[''] # Get features from last layer of backbone

        elif isinstance(Xfeatures, list):   
            X = Xfeatures[-1] # Get features from last layer of backbone
        elif isinstance(Xfeatures, torch.Tensor):
            X = Xfeatures
        else:
            raise ValueError("Input type not supported")
        
        return super().forward(X) 
        
class MultiHead_CentroidRange_V2(MultiHeadRegressor):
    # Centroid head: straight DNN from last layer of backbone
    # Model using features from last layer after adaptive pooling fused with first layer output
    def __init__(self, heads: Union[nn.ModuleList, nn.ModuleDict, nn.Module, dict]):
        super().__init__(heads, enumMultiHeadOutMode.Concatenate)

        self.adaptiveAvgPool_size24 = nn.AdaptiveAvgPool2d(output_size=24) # Define adaptive pooling layer to reduce size of last layer output 
        self.adaptiveAvgPool_size1 = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, Xfeatures:  Union[list, dict]):

        # Manually extract entries from Xfeatures
        if isinstance(Xfeatures, dict):
            raise NotImplementedError("Not implemented yet")
            X = Xfeatures['']  # Get features from last layer of backbone

        elif isinstance(Xfeatures, list):
            
            # Get features from last layer of backbone
            Xlast_resized = self.adaptiveAvgPool_size24(Xfeatures[-1])
            
            # Manually extract entries from Xfeatures
            # Use features from last layer of backbone
            # Use features from last layer of backbone
            Y_centroid = self.heads[0](
                self.adaptiveAvgPool_size1(Xlast_resized))

            # Use features from first and last layer of backbone
            list_of_features = [*Xfeatures[:-1], Xlast_resized]
            Y_range = self.heads[1](list_of_features)

        else:
            raise ValueError("Input type not supported")

        return torch.cat((Y_centroid, Y_range), 1)
        


class MultiHead_CentroidRange_V3(MultiHeadRegressor):
    # Centroid head: straight DNN from last layer of backbone
    # Range head: Model using features from all layers of the backbone (full-multi scale)
    def __init__(self, model_heads: Union[nn.ModuleList, nn.ModuleDict, nn.Module, dict]):
        super().__init__(model_heads, enumMultiHeadOutMode.Concatenate)

    def forward(self, Xfeatures: list):
        Y_centroid = self.model_heads[0](Xfeatures[-1]) 
        Y_range = self.model_heads[1](Xfeatures) # Use features from first and last layer of backbone
        return torch.cat((Y_centroid, Y_range), 1)


class MultiHead_CentroidRange_BiFPN(MultiHeadRegressor):
    # Centroid head: straight DNN from last layer of backbone
    # Range head: Model using features from all layers of the backbone fused through BiFPN structure
    def __init__(self, model_heads: nn.ModuleList | nn.ModuleDict | nn.Module | dict, output_mode: enumMultiHeadOutMode = enumMultiHeadOutMode.Concatenate, *args, **kwargs):
        super().__init__(model_heads, output_mode, *args, **kwargs)
        raise NotImplementedError("Not implemented yet")
        pass 

def build_multiHeadV1(feature_extractor: nn.Module, resAdapter: nn.Module, headCentroid_config: dict, headRange_config: dict) -> nn.Module:

    module_list = nn.ModuleList()
    module_list.append(TemplateDeepNet(headCentroid_config))
    module_list.append(TemplateDeepNet(headRange_config))

    # Define model
    multihead = MultiHead_CentroidRange_V1(module_list)

    model = nn.Sequential(resAdapter, feature_extractor, multihead)

    return model


# DEVNOTE TODO
class FeatureMerger_tailoredForV2(nn.Module):
    def __init__(self):
        super(FeatureMerger_tailoredForV2, self).__init__()

        self.adaptiveMaxPool112 = nn.AdaptiveMaxPool2d(112)

        # Convolutional block for Layer 1 output processing
        self.convBlock = nn.Sequential(
            nn.Conv2d(kernel_size=7, in_channels=32,
                      out_channels=64, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(kernel_size=5, in_channels=64,
                      out_channels=128, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.Conv2d(kernel_size=5, in_channels=128,
                      out_channels=256, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(kernel_size=3, in_channels=256,
                      out_channels=512, stride=2, padding=0),
            nn.SiLU()
        )

        self.adaptiveMaxPool24 = nn.AdaptiveMaxPool2d(24)
        self.convMerger = nn.Conv2d(
            1280 + 512, 1280, kernel_size=1, stride=1, padding=0)

        self.adaptiveAvgPool1 = nn.AdaptiveMaxPool2d(1)
        self.feature_merger = FeatureMerger_tailoredForV2()  # TODO
        self.headRange = TemplateDeepNet(headRange_config)

    def forward(self, Xfeatures: list):
        x = self.feature_merger(Xfeatures)
        return self.headRange(x)


def build_multiHeadV2(feature_extractor: nn.Module, resAdapter: nn.Module, headCentroid_config: dict, headRange_config: dict) -> nn.Module:

    module_list = nn.ModuleList()
    module_list.append(TemplateDeepNet(headCentroid_config))
    range_head = MultiScaleRangeRegressor(headRange_config)

    module_list.append(range_head)  # TODO: Implement range head for V2

    # Define multihead model
    multihead = MultiHead_CentroidRange_V2(module_list)
    model = nn.Sequential(resAdapter, feature_extractor, multihead)

    return model


# %% TEMPORARY DEV
# TODO: make this function generic!
def ReloadModelFromOptuna(trial: optuna.trial.FrozenTrial, other_params: dict, modelName: str, filepath: str) -> nn.Module:

    num_of_epochs = 125
    # other_params = dict()

    # Sample decision parameters space
    # Optimization strategy
    initial_lr = trial.suggest_float('initial_lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 2, 40, step=8)

    # initial_lr = 5e-4
    # other_params['initial_lr'] = initial_lr
    # batch_size = 20
    # other_params['batch_size'] = batch_size

    # Model
    # use_default_size = trial.suggest_int('use_default_size', 0, 1)
    # efficient_net_ID = trial.suggest_int('efficient_net_ID', 0, 1)

    try:
        regressor_arch_version = trial.suggest_int(
            'regressor_arch_version', 1, 2)
    except:
        other_params['regressor_arch_version'] = 1
        regressor_arch_version = other_params['regressor_arch_version']

    # dropout_coeff_multiplier = trial.suggest_int('dropout_coeff_multiplier', 0, 10, step=1)
    dropout_coeff_multiplier = 0
    use_batchnorm = 0

    try:
        image_adapter_strategy = trial.suggest_categorical(
            'image_adapter_strategy', ['resize_copy', 'conv_adapter'])
    except:
        other_params['image_adapter_strategy'] = 'resize_copy'
        image_adapter_strategy = other_params['image_adapter_strategy']

    try:
        mutate_to_groupnorm = trial.suggest_int('mutate_to_groupnorm', 0, 1)
    except:
        other_params['mutate_to_groupnorm'] = 1
        mutate_to_groupnorm = other_params['mutate_to_groupnorm']

    loss_type = trial.suggest_categorical('loss_type', ['mse', 'huber'])

    # use_batchnorm = trial.suggest_int('use_batchnorm', 0, 1)
    num_of_regressor_layers_H1 = trial.suggest_int(
        'num_of_regressor_layers_H1', 3, 7)
    num_of_regressor_layers_H2 = trial.suggest_int(
        'num_of_regressor_layers_H2', 3, 7)

    scheduler = trial.suggest_categorical(
        'lr_scheduler_name', ['cosine_annealing_restarts', 'exponential_decay'])

    if scheduler == 'cosine_annealing_restarts':
        T0_WarmAnnealer = trial.suggest_int('T0_WarmAnnealer', np.floor(
            0.85 * num_of_epochs), 3*num_of_epochs, step=5)
        lr_min = trial.suggest_float('lr_min', 1e-8, 1e-5, log=True)

    elif scheduler == 'exponential_decay':
        gamma = trial.suggest_float('gamma', 0.900, 0.999, step=0.005)

    # Define regressor architecture for centroid prediction
    out_channels_sizes_H1 = []
    out_channels_sizes_H1.append(2)  # Output layer

    for i in range(num_of_regressor_layers_H1):
        out_channels_sizes_H1.append(2**(i+5))

    out_channels_sizes_H1.reverse()
    other_params['out_channels_sizes_H1'] = out_channels_sizes_H1

    out_channels_sizes_H2 = []
    out_channels_sizes_H2.append(2)  # Output layer

    # Define regressor architecture for range prediction
    for i in range(num_of_regressor_layers_H2):
        out_channels_sizes_H2.append(2**(i+5))

    out_channels_sizes_H2.reverse()
    other_params['out_channels_sizes_H2'] = out_channels_sizes_H2

    # Build regression layers with decreasing number of channels as powers of 2
    other_params['model_definition_mode'] = 'multihead'
    model_definition_mode = other_params['model_definition_mode']

    # Print the parameters
    print(f"Parameters: \n"
          f"initial_lr: {initial_lr}\n"
          f"batch_size: {batch_size}\n"
          f"dropout_coeff_multiplier: {dropout_coeff_multiplier}\n"
          f"use_batchnorm: {use_batchnorm}\n"
          f"mutate_to_groupnorm: {mutate_to_groupnorm}\n"
          f"loss_type: {loss_type}\n"
          f"num_of_regressor_layers_H1: {num_of_regressor_layers_H1}\n"
          f"num_of_regressor_layers_H2: {num_of_regressor_layers_H2}\n"
          f"lr_scheduler_name: {scheduler}\n"
          f"out_channels_sizes_H1: {out_channels_sizes_H1}\n"
          f"out_channels_sizes_H2: {out_channels_sizes_H2}\n"
          f"model_definition_mode: {model_definition_mode}\n"
          f"image_adapter_strategy: {image_adapter_strategy}\n"
          f"regressor_arch_version: {regressor_arch_version}\n")

    if scheduler == 'cosine_annealing_restarts':
        print(f"T0_WarmAnnealer: {T0_WarmAnnealer}\n"
              f"lr_min: {lr_min}\n")
    elif scheduler == 'exponential_decay':
        print(f"gamma: {gamma}\n")

    # Define model
    model = DefineModel(trial, other_params)

    # Load model parameters
    model = LoadTorchModel(model, os.path.join(filepath, modelName), False)

    # Loading validation
    validateDictLoading(model, modelName, filepath)

    return model


def validateDictLoading(model: nn.Module | nn.ModuleDict | nn.ModuleList, modelName: str, filepath: str):

    # Load the saved state dict (just to compare)
    checkpoint = torch.load(os.path.join(filepath, modelName+'.pth'))
    saved_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Get the current state dict from the model
    current_state_dict = model.state_dict()

    # Check if the model's parameters match the saved parameters
    for param_name in current_state_dict:
        if not torch.equal(current_state_dict[param_name], saved_state_dict[param_name]):
            raise ValueError(f"Mismatch found in parameter: {param_name}")

    else:
        print("All model parameters are correctly loaded.")


def DefineModel(trial: optuna.trial.Trial | optuna.trial.FrozenTrial, other_params: dict):

    # Decision parameters
    # use_default_size = trial.params['use_default_size']
    if 'efficient_net_ID' in trial.params.keys():
        efficient_net_ID = trial.params['efficient_net_ID']
    else:
        efficient_net_ID = 0

    if efficient_net_ID == 0:
        input_size = 224
    elif efficient_net_ID == 1:
        input_size = 240

    if 'input_size' in other_params.keys():
        input_size = other_params['input_size']
        print(f"Overriding backbone default input size. Using:", input_size)

    # Define EfficientNet backbone output size
    efficientNet_output_size = 1280
    # DEVNOTE: EfficientNet architecture has 1280 feature maps from the last Conv2dNormActivation block.
    # The adaptive average pooling synthesizes each feature map in a single value to avoid need for flattening or reshaping.

    regressor_arch_version = trial.params['regressor_arch_version'] if 'regressor_arch_version' in trial.params.keys(
    ) else other_params['regressor_arch_version']

    if 'model_definition_mode' in other_params.keys():
        model_definition = other_params['model_definition_mode']

        if model_definition == 'multihead' and regressor_arch_version == 1:
            output_type = 'last'
        elif model_definition == 'multihead' and regressor_arch_version == 2:
            output_type = 'features'
        elif model_definition == 'centroid_only':
            output_type = 'last'

    else:
        model_definition = 'centroid_only'
        output_type = ' last'

    feature_extractor = EfficientNetBackbone(
        efficient_net_ID, output_type=output_type)

    if 'dropout_probability' in trial.params.keys():
        dropout_coeff = trial.params['dropout_probability']
    else:
        dropout_coeff = 0

    if 'use_batchnorm' in trial.params.keys():
        use_batchnorm = trial.params['use_batchnorm']
    else:
        use_batchnorm = 0

    if "image_adapter_strategy" in trial.params.keys():
        image_adapter_strategy = trial.params["image_adapter_strategy"]
    elif "image_adapter_strategy" in other_params.keys():
        image_adapter_strategy = other_params["image_adapter_strategy"]
    else:
        image_adapter_strategy = 'conv_adapter'

    # Define adapter model to bring resolution down to feature_extractor input size
    if image_adapter_strategy == 'conv_adapter':

        if 'mutate_to_groupnorm' in trial.params.keys():
            if trial.params['mutate_to_groupnorm'] == 1:
                feature_extractor = (ModelMutator(
                    feature_extractor, 32)).mutate()

        resAdapter = conv2dResolutionAdapter([input_size, input_size], [1, 3])

    elif image_adapter_strategy == 'resize_copy':

        # if 'mutate_to_groupnorm' in trial.params.keys():
        #    if trial.params['mutate_to_groupnorm'] == 1:
        #        mlflow.start_run() # Start new run to log that it is invalid
        #        mlflow.log_param("invalid_trial", True)
        #        mlflow.log_param("image_adapter_strategy", image_adapter_strategy)
        #        mlflow.log_param("mutate_to_groupnorm", trial.params['mutate_to_groupnorm'])
        #        mlflow.end_run(status="KILLED")
        #        optuna.TrialPruned() # Prevent trial to be executed (Combination of parameters is not valid). Batch norm statistics must be preserved.

        if 'mutate_to_groupnorm' in trial.params.keys():
            if trial.params['mutate_to_groupnorm'] == 1:
                feature_extractor = (ModelMutator(
                    feature_extractor, 32)).mutate()

        resAdapter = resizeCopyAdapter(output_size=[input_size, input_size], num_channels=[
                                       1, 3], interp_method='bicubic')  # ACHTUNG: trilinear is for volumetric data

        # Freeze EfficientNet backbone in this strategy
        # feature_extractor.requires_grad_(False)

    # NOTE: expected input size is 240x240, 3 channels
    # Expected output for EfficientNet-B1 feature extractor, the shape of the extracted features before the final classification layer
    # will be approximately (batch_size, 1280, 7, 7) (which represents the spatial dimensions and channels of the last convolutional block)
    # NOTE: but... does using only last layer output implies using only higher level features?
    # print("EfficientNet B model: \n", feature_extractor)

    if 'model_definition_mode' in other_params.keys():
        model_definition = other_params['model_definition_mode']
    else:
        model_definition = 'centroid_only'

    if model_definition == 'centroid_only':
        out_channels_sizes = other_params['out_channels_sizes']

        headCentroid_config = {
            'input_size': efficientNet_output_size,
            'useBatchNorm': use_batchnorm,
            'alphaDropCoeff': dropout_coeff,
            'alphaLeaky': 0.0,
            'outChannelsSizes': out_channels_sizes
        }

        model = nn.Sequential(resAdapter, feature_extractor,
                              TemplateDeepNet(headCentroid_config))
        return model

    elif model_definition == 'multihead':

        out_channels_sizes_H1 = other_params['out_channels_sizes_H1']
        out_channels_sizes_H2 = other_params['out_channels_sizes_H2']

        headCentroid_config = {
            'input_size': efficientNet_output_size,
            'useBatchNorm': use_batchnorm,
            'alphaDropCoeff': dropout_coeff,
            'alphaLeaky': 0.0,
            'outChannelsSizes': out_channels_sizes_H1
        }

        # Define Regression head
        headRange_config = {
            'input_size': efficientNet_output_size,
            'useBatchNorm': use_batchnorm,
            'alphaDropCoeff': dropout_coeff,
            'alphaLeaky': 0.0,
            'outChannelsSizes': out_channels_sizes_H2
        }

        if regressor_arch_version == 1:
            model = build_multiHeadV1(
                feature_extractor, resAdapter, headCentroid_config, headRange_config)

        elif regressor_arch_version == 2:
            model = build_multiHeadV2(
                feature_extractor, resAdapter, headCentroid_config, headRange_config)

        return model
