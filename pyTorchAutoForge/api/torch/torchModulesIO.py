import torch, sys, os
from torch.utils.data import Dataset
import torch.utils

def SaveTorchModel(model: torch.nn.Module, modelName: str = "trainedModel", saveAsTraced: bool = False, exampleInput=None, targetDevice: str = 'cpu') -> None:
    if 'os.path' not in sys.modules:
        import os.path

    if saveAsTraced:
        extension = '.pt'
    else:
        extension = '.pth'

    # Format target device string to remove ':' from name
    targetDeviceName = targetDevice
    targetDeviceName = targetDeviceName.replace(':', '')

    if modelName == 'trainedModel':
        if not (os.path.isdir('./testModels')):
            os.mkdir('testModels')
            if not (os.path.isfile('.gitignore')):
                # Write gitignore in the current folder if it does not exist
                gitignoreFile = open('.gitignore', 'w')
                gitignoreFile.write("\ntestModels/*")
                gitignoreFile.close()
            else:
                # Append to gitignore if it exists
                gitignoreFile = open('.gitignore', 'a')
                gitignoreFile.write("\ntestModels/*")
                gitignoreFile.close()

        filename = "testModels/" + modelName + '_' + targetDeviceName + extension
    else:
        filename = modelName + '_' + targetDeviceName + extension

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
def LoadTorchModel(model: torch.nn.Module = None, modelName: str = "trainedModel", filepath: str = "testModels/", loadAsTraced: bool = False) -> torch.nn.Module:

    # Check if input name has extension
    modelNameCheck, extension = os.path.splitext(str(modelName))

    # print(modelName, ' ', modelNameCheck, ' ', extension)

    if extension != '.pt' and extension != '.pth':
        if loadAsTraced:
            extension = '.pt'
        else:
            extension = '.pth'
    else:
        extension = ''

    # Contatenate file path
    modelPath = os.path.join(filepath, modelName + extension)

    if not (os.path.isfile(modelPath)):
        raise FileNotFoundError('Model specified by: ',
                                modelPath, ': NOT FOUND.')

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
