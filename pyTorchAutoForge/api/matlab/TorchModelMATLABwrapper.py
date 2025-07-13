from pickle import NONE
from pyTorchAutoForge.utils.utils import GetDevice
from pyTorchAutoForge.model_building.modelBuildingBlocks import AutoForgeModule
from pyTorchAutoForge.api.torch import LoadModel
import numpy as np
import torch, os
from torch import nn
from dataclasses import dataclass

@dataclass 
class MatlabWrapperConfig():
    # Default wrapper configuration
    DEBUG_MODE: bool = False
    device : torch.device | str | None = None
    input_shape_validation: list | None = None # None for no validation
    loading_mode: str | None = None # 'traced', 'state_dict', None
    trained_model_path: str | None = None
    

# %% MATLAB wrapper class for Torch models evaluation - 11-06-2024 # TODO: update class
class TorchModelMATLABwrapper():
    '''Class to wrap a trained PyTorch model for evaluation in MATLAB'''
    def __init__(self, trainedModel: str | nn.Module | AutoForgeModule, 
                 wrapperConfig: MatlabWrapperConfig = MatlabWrapperConfig(),
                 modelArch: nn.Module | None = None) -> None:
        
        '''Constructor for TorchModelMATLABwrapper'''

        if wrapperConfig == MatlabWrapperConfig():
            print('Using default wrapper configuration...')

        # Initialize using configuration class
        self.DEBUG_MODE = wrapperConfig.DEBUG_MODE
        self.device = wrapperConfig.device if wrapperConfig.device is not None else GetDevice()
        self.enable_warning = True # To disable warning for batch size
        self.input_shape_validation = wrapperConfig.input_shape_validation
        self.loading_mode = wrapperConfig.loading_mode
        self.trained_model_path = wrapperConfig.trained_model_path

        # Define flag for model loading mode
        traced_loading = True if self.loading_mode.lower() == 'traced' else False
        
        # Check modelArch is provided if state_dict loading mode is selected
        #if self.loading_mode.lower() == 'state_dict' and modelArch is None:
        #    raise ValueError(
        #        'Model architecture must be provided for state_dict loading mode. Please provide modelArch as nn.Module or callable function with modelArch as output.')

        if isinstance(trainedModel, (nn.Module, AutoForgeModule)) and self.loading_mode is None:
            self.trainedModel = trainedModel # Assume model is already loaded and provided

        elif isinstance(trainedModel, (nn.Module, AutoForgeModule)) and self.loading_mode == 'traced':
            self.trainedModel = LoadModel(
                None, self.trained_model_path, loadAsTraced=traced_loading).to(self.device)

        elif isinstance(trainedModel, (nn.Module, AutoForgeModule)) and self.loading_mode == 'state_dict':
            self.trainedModel = LoadModel(
                trainedModel, self.trained_model_path, loadAsTraced=traced_loading).to(self.device)
        else:
            raise ValueError(
                'Invalid input for trainedModel, trained_model_path and loading_mode or invalid combination. Please provide a valid model path or nn.Module object.')
        
        (self.trainedModel).eval()

        # Print model data
        if self.DEBUG_MODE:
            print("The following model has been loaded and will be used in forward() call: \n", self.trainedModel)


    def forward(self, inputSample: np.ndarray | torch.Tensor, numBatches: int | None = None, inputShape: list[int] | None = None) -> np.ndarray:
        '''Forward method to perform inference on N sample input using loaded trainedModel. Batch size assumed as 1 if not given.'''
        
        if self.DEBUG_MODE:
            print('Input sample shape: ', inputSample.shape, 'on device: ', self.device)
        
        if self.input_shape_validation is not None:
            if inputSample.shape != self.input_shape_validation:
                raise ValueError(f'Input shape {inputSample.shape} does not match the expected shape: {self.input_shape_validation}')
            
        try:
            # Set batch size if not provided
            if numBatches != None and len(inputSample.shape) > 2:
                # Assume first dimension of array is batch size
                numBatches = inputSample.shape[0]
            else:
                if self.enable_warning:
                    Warning('Batch size not provided and input is two-dimensional. Assuming batch size of 1.')
                    self.enable_warning = False
                numBatches = 1

            # Check input type and convert to torch.tensor if necessary
            if isinstance(inputSample, np.ndarray) and inputSample.dtype != np.float32:
                Warning('Converting input to np.float32 from', inputSample.dtype)
                inputSample = torch.from_numpy(np.float32(inputSample))

            elif isinstance(inputSample, torch.Tensor) and inputSample.dtype != torch.float32:
                Warning('Converting input to torch.float32 from', inputSample.dtype)
                inputSample = inputSample.float()


            if inputShape is not None:
                raise NotImplementedError('Input shape validation/reshaping is not implemented yet. Please set inputShape to None for now.')
                X = inputSample.view()
            else:
                # Reshape according to batch dimension
                X = inputSample

        except Exception as e:
            max_chars = 400  # Define the max length you want to print
            print( f"\nError during input preprocessing: {str(e)[:max_chars]}...")
            return str(e)[:max_chars]
        
        # Perform inference using model
        try:
            Y = self.trainedModel(X.to(self.device))
        except Exception as e:        
            max_chars = 400  # Define the max length you want to print
            print(f"\nError during model inference: {str(e)[:max_chars]}...")
            return str(e)[:max_chars]

        # ########### DEBUG ######################:
        if self.DEBUG_MODE:
            print('Model prediction: ', Y)
        ############################################

        return Y.detach().cpu().numpy()  # Move to cpu and convert to numpy before returning

    def ValidateModelPath(self, trainedModel):
        # Load model as traced
        if isinstance(trainedModel, str):

            # Get extension of model file
            filepath_noext, extension = os.path.splitext(str(trainedModel))

            # Check if extension is provided
            if extension != '.pt' and extension == '.pth':
                raise ValueError(
                    'Please provide a .pt file. This function only supports traced models at current stage and cannot load .pth state dict.')

            elif extension != '.pt' and extension == '' and self.loading_mode.lower() == 'traced':
                print(
                    'No extension provided. Assuming .pt extension for model file (traced).')
                self.trained_model_path = trainedModel + ".pt"  # Assume .pt extension

            elif extension != '.pt' and extension == '' and self.loading_mode.lower() == 'state_dict':
                print(
                    'No extension provided. Assuming .pth extension for model file (state_dict).')
                self.trained_model_path = trainedModel + ".pth"  # Assume .pth extension

            elif (extension == '.pt' and self.loading_mode.lower() == 'traced') or (extension == '.pth' and self.loading_mode.lower() == 'state_dict'):
                self.trained_model_path = trainedModel

            else:
                raise ValueError(
                    'Invalid configuration: provided extesion does not match loading_mode configuration. Valid cases: .pt with loading_mode=traced, .pth with loading_mode=state_dict')


def test_TorchModelMATLABwrapper():
    # Get script path
    import os
    file_dir = os.path.dirname(os.path.realpath(__file__))

    module_path = os.path.join('/home/peterc/devDir/pyTorchAutoForge/tests/data/sample_cnn_traced_cpu')
    print(module_path)

    # Check if model exists
    if not os.path.isfile(module_path + '.pt'):
        raise FileNotFoundError('Model specified by: ',
                                module_path, ': NOT FOUND. Run create_sample_tracedModel.py to create model first.')

    # Define wrapper configuration and wrapper object
    wrapper_config = MatlabWrapperConfig()

    model_wrapper = TorchModelMATLABwrapper(module_path, wrapper_config)

    # Test forward method
    input_sample = np.random.rand(1, 3, 256, 256)

    print('Input shape:', input_sample.shape)
    output = model_wrapper.forward(input_sample)
    assert isinstance(output, np.ndarray) 
    assert output.shape == (1, 16, 252, 252)
    print('Output shape:', output.shape)
    

if __name__ == '__main__':
    test_TorchModelMATLABwrapper()