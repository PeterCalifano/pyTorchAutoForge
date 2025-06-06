"""
    Module containing a set of basic functions to load and save objects inheriting nn.Module (models and datasets).
"""

from email.policy import strict
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
    TRACED_DINAMO = "traced_dynamo"
    SCRIPTED_TORCHSCRIPT = "scripted_torchscript"
    MODEL_STATE_DICT = "model_state_dict"
    MODEL_ARCH_STATE = "model_arch_state"


def SaveModel(model: torch.nn.Module, 
              model_filename: str | pathlib.Path, 
              save_mode : AutoForgeModuleSaveMode | str = AutoForgeModuleSaveMode.MODEL_ARCH_STATE, 
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
            Defaults to AutoForgeModuleSaveMode.MODEL_ARCH_STATE.
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

    elif save_mode == AutoForgeModuleSaveMode.TRACED_DINAMO or save_mode == AutoForgeModuleSaveMode.SCRIPTED_TORCHSCRIPT:
        print('Warning: tracing/scripting requested, but no sample input was provided. Defaulting to save model without.')

    if save_mode == AutoForgeModuleSaveMode.TRACED_DINAMO or save_mode == AutoForgeModuleSaveMode.SCRIPTED_TORCHSCRIPT and example_input is not None:

        extension = '.pt'
        traced_or_scripted = True

    elif save_mode == AutoForgeModuleSaveMode.MODEL_STATE_DICT:
        extension = '_statedict.pth'

    else: 
        extension = '.pth'
        
    # Format target device string to remove ':' from name
    target_device_name = target_device
    target_device_name = target_device_name.replace(':', '')

    # Form filename for saving
    # Check if device is in model name and remove it
    if save_mode == AutoForgeModuleSaveMode.TRACED_DINAMO or save_mode == AutoForgeModuleSaveMode.SCRIPTED_TORCHSCRIPT:

        # Append device for which the traced model was saved on
        if ("_" + target_device_name) in str(model_filename):
            model_filename = str(model_filename).replace("_" + target_device_name, "")

        model_filename = model_filename + "_" + target_device_name

    model_filename = model_filename + extension

    # Get directory from modelpath and check it exists
    saving_dir = os.path.dirname(model_filename)
    os.makedirs(saving_dir, exist_ok=True)

    model.eval()
    if traced_or_scripted == True and example_input is not None:
        
        if save_mode == AutoForgeModuleSaveMode.TRACED_DINAMO:
            from torch.export import export, save
            exported_program = export(model.to(target_device), (example_input.to(target_device),))
            save(exported_program, model_filename)

            print("Saving traced_dynamo torch model as file:", model_filename)

        elif save_mode == AutoForgeModuleSaveMode.SCRIPTED_TORCHSCRIPT:
            
            traced_model = torch.jit.trace(model.to(target_device), example_input.to(target_device))
            torch.jit.save(traced_model, model_filename)
            print("Saving scripted_torchscript torch model as file:", model_filename)

        else:
            raise ValueError('Invalid save mode for traced model. Valid options: traced_dynamo, scripted_torchscript')
        
        return
    
    elif traced_or_scripted == False:
        
        if save_mode == AutoForgeModuleSaveMode.MODEL_STATE_DICT:
            
            print("Saving state dict of torch model as file:", model_filename)
            # Save model as internal torch representation
            torch.save(model.state_dict(), model_filename)

        else:

            print("Saving torch model as file:", model_filename)
            # Save model as internal torch representation
            torch.save(model, model_filename)

        return


# %% Function to load model state into empty model- 04-05-2024, updated 11-06-2024
def LoadModel(model: torch.nn.Module | None, 
              model_filename: str, 
              load_as_traced: bool = False, 
              load_strict : bool = False) -> torch.nn.Module:

    # Check if input name has extension
    modelNameCheck, extension = os.path.splitext(str(model_filename))

    # TODO improve this section of LoadModel using enumeration class
    if extension != '.pt' and extension != '.pth':
        if load_as_traced:
            extension = '.pt'
        else:
            extension = '.pth'
    else:
        extension = ''

    # Contatenate file path
    model_filepath = model_filename + extension

    if not (os.path.isfile(model_filepath)):
        raise FileNotFoundError('No file found at:', model_filepath)

    if load_as_traced:
        print('Loading traced model from filename: ', model_filepath)
        # Load traced model using torch.jit

        if model is not None:
            print('\033[38;5;208mload_as_traced is specified as true, but model has been provided. Model will be overwritten by checkpoint load.\033[0m')

        model = torch.jit.load(model_filepath)

        if model is None:
            raise ValueError('Model loaded from file failed: found to be None.')
        
        print('Traced model correctly loaded.')

        return model.eval()
    
    else:
        print(f'Loading model, load_strict={load_strict}, from file: {model_filepath}')

        if model is not None:
            print("Selected mode: 'state dict only'.")
            loaded_model_ = torch.load(model_filepath, map_location="cpu", weights_only=False)

            # Handle different input formats (state dict only, model with state dict)
            if hasattr(loaded_model_, "state_dict"):
                loaded_state = loaded_model_.state_dict()

            elif isinstance(loaded_model_, dict):
                # If dict, check for state_dict or model_state_dict keys
                if "state_dict" in loaded_model_:
                    loaded_state = loaded_model_["state_dict"]
                elif "model_state_dict" in loaded_model_:
                    loaded_state = loaded_model_["model_state_dict"]
                else:
                    # If loaded_model_ is actually a state dict itself
                    loaded_state = loaded_model_
            else:
                raise ValueError("Unrecognized checkpoint format")
            
            # Inject state dict into model
            incompatible_keys = model.load_state_dict(loaded_state, strict=load_strict)

            if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
                if incompatible_keys.missing_keys:
                    print(f'\033[38;5;208mWarning: Missing keys in state_dict: {incompatible_keys.missing_keys}\033[0m')
                if incompatible_keys.unexpected_keys:
                    print(f'\033[38;5;214mWarning: Unexpected keys in state_dict: {incompatible_keys.unexpected_keys}\033[0m')

                # Wait for user input to continue
                usr_input = input('Do you want to continue anyway? (Y/n): ')

                while usr_input.lower() not in ['y', 'n', 'no', 'yes']:
                    usr_input = input('Please enter Y or n: ').lower()

                if usr_input == 'n' or usr_input == 'no':
                    print('Termination input received. Exiting program...')
                    sys.exit(0)
                elif usr_input == 'Y' or usr_input == 'yes':
                    print('Continuation input received. Going on with program execution...')
                
        else:
            print("Selected mode: 'model with state dict'.")
            model = torch.load(model_filepath, 
                               map_location='cpu', 
                               weights_only=False)
            
        if model is None:
            raise ValueError('Model loaded failed: found to be None.')
        print('Model correctly loaded.')
        return model.eval()

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


# %% Auxiliary functions

def ValidateDictLoading(model: torch.nn.Module | torch.nn.ModuleDict | torch.nn.ModuleList,
                        model_name: str, 
                        filepath: str):

    # Load the saved state dict (just to compare)
    checkpoint = torch.load(os.path.join(filepath, model_name +'.pth'))
    saved_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Get the current state dict from the model
    current_state_dict = model.state_dict()

    # Check if the model's parameters match the saved parameters
    for param_name in current_state_dict:
        if not torch.equal(current_state_dict[param_name], saved_state_dict[param_name]):
            raise ValueError(f"Mismatch found in parameter: {param_name}")

    else:
        print("All model parameters are correctly loaded.")


