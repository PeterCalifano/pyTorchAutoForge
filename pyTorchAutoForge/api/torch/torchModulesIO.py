"""
    Module containing a set of basic functions to load and save objects inheriting nn.Module (models and datasets).
"""

from enum import Enum
from numpy import deprecate
import torch, sys, os
from torch.utils.data import Dataset
from pyTorchAutoForge.utils.utils import AddZerosPadding


class AutoForgeModuleSaveMode(Enum):
    """   
    Enumeration for AutoForge Module Save Modes.

    This enum defines the various methods available for saving modules,
    including approaches that use tracing and state dictionary management.

    Attributes:
        traced_dynamo (str): Save the module using the traced dynamo approach.
        traced_torchscript (str): Save the module using the traced TorchScript method.
        model_state_dict (str): Save the module's state dictionary.
        model_arch_state (str): Save the model's architecture state.
    """
    traced_dynamo = "traced_dynamo"
    traced_torchscript = "traced_torchscript"
    model_state_dict = "model_state_dict"
    model_arch_state = "model_arch_state"


def SaveModel(model: torch.nn.Module, modelpath: str = "./trainedModel", saveAsTraced: bool = False, weightsOnly : bool = False, exampleInput: torch.Tensor | None = None, targetDevice: str = 'cpu') -> None:
    
    # Determine extension
    if saveAsTraced:
        # Overrides everything else
        extension = '.pt'

    elif weightsOnly:
        # State dict only
        extension = '_stateDict.pth'

    else: 
        # Default extension
        extension = '.pth'

    if exampleInput is not None:
        exampleInput = exampleInput.detach()
        exampleInput.requires_grad = False
        
    # Format target device string to remove ':' from name
    targetDeviceName = targetDevice
    targetDeviceName = targetDeviceName.replace(':', '')

    # Check if device is in model name and remove it
    if ("_" + targetDeviceName) in modelpath:
        modelpath = modelpath.replace("_" + targetDeviceName, '')

    if modelpath == 'trainedModel':
        if not (os.path.isdir('./savedModels')):
            os.mkdir('savedModels')
            if not (os.path.isfile('.gitignore')):
                # Write gitignore in the current folder if it does not exist
                gitignoreFile = open('.gitignore', 'w')
                gitignoreFile.write("\nsavedModels/*")
                gitignoreFile.close()
            else:
                # Append to gitignore if it exists
                gitignoreFile = open('.gitignore', 'a')
                gitignoreFile.write("\nsavedModels/*")
                gitignoreFile.close()

        filename = "savedModels/" + modelpath + '_' + targetDeviceName + extension
    else:
        filename = modelpath + '_' + targetDeviceName + extension

        # Get directory from modelpath and check it exists
        modelpath_only = os.path.dirname(filename)
        if not (os.path.isdir(modelpath_only)):
            os.makedirs(modelpath_only)

    # Attach timetag to model checkpoint
    # currentTime = datetime.datetime.now()
    # formattedTimestamp = currentTime.strftime('%d-%m-%Y_%H-%M') # Format time stamp as day, month, year, hour and minute

    # filename =  filename + "_" + formattedTimestamp
    print("Saving PyTorch Model State to:", filename)

    if saveAsTraced:
        print('Saving traced model...')
        if exampleInput is not None:
            tracedModel = torch.jit.trace((model).to(
                targetDevice), exampleInput.to(targetDevice))
            tracedModel.save(filename)
            print('Model correctly saved with filename: ', filename)
        else:
            raise ValueError(
                'You must provide an example input to trace the model through torch.jit.trace()')
    else:
        print('Saving NOT traced model...')
        # Save model as internal torch representation
        torch.save(model.state_dict(), filename)
        print('Model correctly saved with filename: ', filename)


# %% Function to load model state into empty model- 04-05-2024, updated 11-06-2024
def LoadModel(model: torch.nn.Module = None, modelpath: str = "savedModels/trainedModel.pt", loadAsTraced: bool = False) -> torch.nn.Module:

    # Check if input name has extension
    modelNameCheck, extension = os.path.splitext(str(modelpath))

    # print(modelName, ' ', modelNameCheck, ' ', extension)

    if extension != '.pt' and extension != '.pth':
        if loadAsTraced:
            extension = '.pt'
        else:
            extension = '.pth'
    else:
        extension = ''

    # Contatenate file path
    modelPath = modelpath + extension

    if not (os.path.isfile(modelPath)):
        raise FileNotFoundError('No file found at:', modelPath)

    if loadAsTraced and model is None:
        print('Loading traced model from filename: ', modelPath)
        # Load traced model using torch.jit
        model = torch.jit.load(modelPath)
        print('Traced model correctly loaded.')

    elif not (loadAsTraced) or (loadAsTraced and model is not None):

        if loadAsTraced and model is not None:
            print('loadAsTraced is specified as true, but model has been provided. Loading from state: ', modelPath)
        else:
            print('Loading model from filename: ', modelPath)

        # Load model from file
        model.load_state_dict(torch.load(modelPath))
        # Evaluate model to set (weights, biases)
        model.eval()

    else:
        raise ValueError('Incorrect combination of inputs! Valid options: \n  1) model is None AND loadAsTraced is True; \n  2) model is nn.Module AND loadAsTraced is False; \n  3) model is nn.Module AND loadAsTraced is True (fallback to case 2)')

    return model


# %% Function to save Dataset object - 01-06-2024
def SaveDataset(datasetObj: Dataset, datasetFilePath: str = '', datasetName: str = 'dataset') -> None:

    try:
        if not (os.path.isdir(datasetFilePath)):
            os.makedirs(datasetFilePath)
        torch.save(datasetObj, os.path.join(
            datasetFilePath, datasetName + ".pt"))
    except Exception as exception:
        print('Failed to save dataset object with error: ', exception)

# %% Function to load Dataset object - 01-06-2024
def LoadDataset(datasetFilePath: str, datasetName: str = 'dataset') -> Dataset:
    return torch.load(os.path.join(datasetFilePath, datasetName + ".pt"))

