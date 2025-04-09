"""
    Module containing a set of basic functions to load and save objects inheriting nn.Module (models and datasets).
"""

from enum import Enum
from matplotlib.patches import Patch
from onnx import save
import torch, sys, os
from torch.utils.data import Dataset
from zipp import Path
from pyTorchAutoForge.utils.utils import AddZerosPadding
import pathlib


class AutoForgeModuleSaveMode(Enum):
    """   
    Enumeration for AutoForge Module Save Modes.

    This enum defines the various methods available for saving modules,
    including approaches that use tracing and state dictionary management.

    Attributes:
        traced_dynamo (str): Save the module using the traced dynamo approach.
        scripted_torchscript (str): Save the module using the traced TorchScript method.
        model_state_dict (str): Save the module's state dictionary.
        model_arch_state (str): Save the model's architecture state.
    """
    traced_dynamo = "traced_dynamo"
    scripted_torchscript = "scripted_torchscript"
    model_state_dict = "model_state_dict"
    model_arch_state = "model_arch_state"


def SaveModel(model: torch.nn.Module, 
              model_filename: str | pathlib.Path, 
              save_mode : AutoForgeModuleSaveMode | str = AutoForgeModuleSaveMode.model_arch_state, 
              example_input: torch.Tensor | None = None, 
              target_device: str = 'cpu', 
              model_base_name : str | None = None) -> None:
    """
    Saves a PyTorch model to a file.

    Depending on the provided save_mode, this function either saves the entire model,
    its state dictionary, or a traced/scripted version of the model. If an example_input
    is provided for tracing/scripted modes, the model is processed accordingly.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        model_filename (str | pathlib.Path): The path or file name to save the model.
        save_mode (AutoForgeModuleSaveMode | str, optional): The mode for saving the model.
            It can be one of the following:
                - traced_dynamo: Save the model using the traced dynamo approach.
                - scripted_torchscript: Save the model using the TorchScript method.
                - model_state_dict: Save only the model's state dictionary.
                - model_arch_state: Save the model's architectural state.
            Defaults to AutoForgeModuleSaveMode.model_arch_state.
        example_input (torch.Tensor | None, optional): A sample input tensor for tracing or scripting.
            Defaults to None.
        target_device (str, optional): The target device (e.g., 'cpu' or 'cuda:0') to save the model.
            Defaults to 'cpu'.
        model_base_name (str | None, optional): An optional base name for the model.
            Defaults to None.

    Raises:
        ValueError: If an invalid save_mode is provided for traced/scripted saving, or if required
            parameters for tracing/scripted saving are missing.
    """

    # Cast modelpath to string
    model_filename = str(model_filename)

    # Stip extension if it exists
    model_filename, _ = os.path.splitext(model_filename)

    # Determine extension if not provided
    traced_or_scripted = False

    if example_input is not None:
        example_input = example_input.detach()
        example_input.requires_grad = False

    elif save_mode == AutoForgeModuleSaveMode.traced_dynamo or save_mode == AutoForgeModuleSaveMode.scripted_torchscript:
        print('Warning: tracing/scripting requested, but no sample input was provided. Defaulting to save model without.')

    if save_mode == AutoForgeModuleSaveMode.traced_dynamo or save_mode == AutoForgeModuleSaveMode.scripted_torchscript and example_input is not None:

        extension = '.pt'
        traced_or_scripted = True

    elif save_mode == AutoForgeModuleSaveMode.model_state_dict:
        extension = '_statedict.pth'

    else: 
        extension = '.pth'
        
    # Format target device string to remove ':' from name
    target_device_name = target_device
    target_device_name = target_device_name.replace(':', '')

    # Form filename for saving
    # Check if device is in model name and remove it
    if save_mode == AutoForgeModuleSaveMode.traced_dynamo or save_mode == AutoForgeModuleSaveMode.scripted_torchscript:

        # Append device for which the traced model was saved on
        if ("_" + target_device_name) in str(model_filename):
            model_filename = str(model_filename).replace("_" + target_device_name, "")

        model_filename = model_filename + "_" + target_device_name

    model_filename = model_filename + extension

    # Get directory from modelpath and check it exists
    saving_dir = os.path.dirname(model_filename)
    os.makedirs(saving_dir, exist_ok=True)

    if traced_or_scripted == True and example_input is not None:
        
        if save_mode == AutoForgeModuleSaveMode.traced_dynamo:

            traced_model = torch._dynamo.export(model.to(target_device), example_input.to(target_device))
            print("Saving traced_dynamo torch model as file:", model_filename)

        elif save_mode == AutoForgeModuleSaveMode.scripted_torchscript:
            
            traced_model = torch.jit.trace(model.to(target_device), example_input.to(target_device))
            print("Saving scripted_torchscript torch model as file:", model_filename)

        else:
            raise ValueError('Invalid save mode for traced model. Valid options: traced_dynamo, scripted_torchscript')
        
        torch.jit.save(traced_model, model_filename) 
        return
    
    elif traced_or_scripted == False:
        
        if save_mode == AutoForgeModuleSaveMode.model_state_dict:
            
            print("Saving state dict of torch model as file:", model_filename)
            # Save model as internal torch representation
            torch.save(model.state_dict(), model_filename)

        else:

            print("Saving torch model as file:", model_filename)
            # Save model as internal torch representation
            torch.save(model, model_filename)

        return


# %% Function to load model state into empty model- 04-05-2024, updated 11-06-2024
def LoadModel(model: torch.nn.Module | None, model_filename: str, loadAsTraced: bool = False) -> torch.nn.Module:

    # Check if input name has extension
    modelNameCheck, extension = os.path.splitext(str(model_filename))

    # print(modelName, ' ', modelNameCheck, ' ', extension)

    if extension != '.pt' and extension != '.pth':
        if loadAsTraced:
            extension = '.pt'
        else:
            extension = '.pth'
    else:
        extension = ''

    # Contatenate file path
    modelPath = model_filename + extension

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

