import torch, sys, os
from torch.utils.data import Dataset
import torch.utils

from pyTorchAutoForge.utils.utils import AddZerosPadding

# TODO: UPDATE FUNCTION
def SaveTorchModel(model: torch.nn.Module, modelpath: str = "./trainedModel", saveAsTraced: bool = False, exampleInput: torch.Tensor = None, targetDevice: str = 'cpu') -> None:
   
    if saveAsTraced:
        extension = '.pt'
    else:
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
def LoadTorchModel(model: torch.nn.Module = None, modelpath: str = "savedModels/trainedModel.pt", loadAsTraced: bool = False) -> torch.nn.Module:

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
def SaveTorchDataset(datasetObj: Dataset, datasetFilePath: str = '', datasetName: str = 'dataset') -> None:

    try:
        if not (os.path.isdir(datasetFilePath)):
            os.makedirs(datasetFilePath)
        torch.save(datasetObj, os.path.join(
            datasetFilePath, datasetName + ".pt"))
    except Exception as exception:
        print('Failed to save dataset object with error: ', exception)

# %% Function to load Dataset object - 01-06-2024
def LoadTorchDataset(datasetFilePath: str, datasetName: str = 'dataset') -> Dataset:
    return torch.load(os.path.join(datasetFilePath, datasetName + ".pt"))

# %% Function to get model checkpoint and load it into nn.Module for training restart - 09-06-2024
def LoadModelAtCheckpoint(model: torch.nn.Module, modelSavePath: str = './checkpoints', 
                          modelName: str = 'trainedModel', modelEpoch: int = 0) -> torch.nn.Module:
    # TODO: add checks that model and checkpoint matches: how to? Check number of parameters?

    # Create path to model state file
    checkPointPath = os.path.join(
        modelSavePath, modelName + '_' + AddZerosPadding(modelEpoch, stringLength=4))

    # Attempt to load the model state and evaluate it
    if os.path.isfile(checkPointPath):
        print('Loading model to RESTART training from checkpoint: ', checkPointPath)
        try:
            loadedModel = LoadTorchModel(model, modelName, modelSavePath)
        except Exception as exception:
            print('Loading of model for training restart failed with error:', exception)
            print('Skipping reload and training from scratch...')
            return model
    else:
        raise ValueError(
            'Specified model state file not found. Check input path.')

    # Get last saving of model (NOTE: getmtime does not work properly. Use scandir + list comprehension)
    # with os.scandir(modelSavePath) as it:
        # modelNamesWithTime = [(entry.name, entry.stat().st_mtime) for entry in it if entry.is_file()]
        # modelName = sorted(modelNamesWithTime, key=lambda x: x[1])[-1][0]

    return loadedModel
