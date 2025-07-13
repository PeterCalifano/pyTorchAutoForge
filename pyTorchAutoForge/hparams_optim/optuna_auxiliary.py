import mlflow, optuna
from torch import nn
from torch.utils.data import DataLoader
from pyTorchAutoForge.utils.utils import GetDevice, AddZerosPadding
from pyTorchAutoForge.optimization.ModelTrainingManager import TrainModel, ValidateModel
from pyTorchAutoForge.api.torch.torchModulesIO import SaveModel, LoadModel
import numpy as np
import os, copy
import torch


class HyperParamOptimManagerConfig():

    def __init__(self) -> None:
        # Suggested by Copilot:
        self.studyName = 'OptunaStudy'
        self.numOfTrials = 100
        self.numOfEpochs = 10
        self.lossFcn = None
        self.optimizer = None
        self.dataloaderIndex = {}
        self.model = None
        self.options = {}

        self.mlflowTrackingURI = 'http://localhost:5000'
        self.mlflowExperimentName = 'DefaultExperiment'
        self.mlflowRunName = 'OptunaRun'
        self.mlflowTags = {}
        self.mlflowParams = {}


class HyperParamOptimManager(HyperParamOptimManagerConfig):
    def __init__(self) -> None:
        super().__init__()

    def optimizeHyperParams(self):
        pass

    def runTrial_(self):
        '''Run for study for hyperparameter'''
        pass


# %% New version of TrainAndValidateModel with MLFlow logging specifically design for Optuna studies - 11-07-2024
def TrainAndValidateModelForOptunaOptim(trial, dataloaderIndex: dict, model: nn.Module, lossFcn: nn.Module, optimizer, options: dict = {}):
    '''Training and validation loop for PyTorch models with MLFlow logging for Optuna optimization studies'''
    # NOTE: Classification is not well developed (July, 2024). Default is regression
    taskType = options.get('taskType', 'regression')
    device = options.get('device', None)
    numOfEpochs = options.get('epochs', 10)
    enableSave = options.get('saveCheckpoints', True)
    checkpointDir = options.get('checkpointsOutDir', './checkpoints')
    modelName = options.get('modelName', 'trainedModel')
    lossLogName = options.get('lossLogName', 'Loss_value')
    epochStart = options.get('epochStart', 0)

    lr_scheduler = options.get('lr_scheduler', None)

    if device is None:
        device = GetDevice()
        
    # if 'enableAddImageToTensorboard' in options.keys():
    #    ADD_IMAGE_TO_TENSORBOARD = options['enableAddImageToTensorboard']
    # else:
    #    ADD_IMAGE_TO_TENSORBOARD = True

    # if ('tensorBoardPortNum' in options.keys()):
    #    tensorBoardPortNum = options['tensorBoardPortNum']
    # else:
    #    tensorBoardPortNum = 6006

    # Get Torch dataloaders
    if ('TrainingDataLoader' in dataloaderIndex.keys() and 'ValidationDataLoader' in dataloaderIndex.keys()):
        trainingDataset = dataloaderIndex['TrainingDataLoader']
        validationDataset = dataloaderIndex['ValidationDataLoader']

        if not (isinstance(trainingDataset, DataLoader)):
            raise TypeError(
                'Training dataloader is not of type "DataLoader". Check configuration.')
        if not (isinstance(validationDataset, DataLoader)):
            raise TypeError(
                'Validation dataloader is not of type "DataLoader". Check configuration.')

    else:
        raise IndexError(
            'Configuration error: either TrainingDataLoader or ValidationDataLoader is not a key of dataloaderIndex')

    # Configure Tensorboard
    # if 'logDirectory' in options.keys():
    #    logDirectory = options['logDirectory']
    # else:
    #    currentTime = datetime.datetime.now()
    #    formattedTimestamp = currentTime.strftime('%d-%m-%Y_%H-%M') # Format time stamp as day, month, year, hour and minute
    #    logDirectory = './tensorboardLog_' + modelName + formattedTimestamp

    # if not(os.path.isdir(logDirectory)):
    #    os.mkdir(logDirectory)
    # tensorBoardWriter = ConfigTensorboardSession(logDirectory, portNum=tensorBoardPortNum)

    # If training is being restarted, attempt to load model
    if options['loadCheckpoint'] == True:
        raise NotImplementedError(
            'Training restart from checkpoint REMOVED. Not updated with mlflow yet.')
        model = LoadModelAtCheckpoint(
            model, options['checkpointsInDir'], modelName, epochStart)

    # Move model to device if possible (check memory)
    try:
        print('Moving model to selected device:', device)
        model = model.to(device)  # Create instance of model using device
    except Exception as exception:
        # Add check on error and error handling if memory insufficient for training on GPU:
        print('Attempt to load model in', device,
              'failed due to error: ', repr(exception))

    # Training and validation loop
    # input('\n-------- PRESS ENTER TO START TRAINING LOOP --------\n')
    trainLossHistory = np.zeros(numOfEpochs)
    validationLossHistory = np.zeros(numOfEpochs)

    numOfUpdates = 0
    prevBestValidationLoss = 1E10
    # Deep copy the initial state of the model and move it to the CPU
    bestModel = copy.deepcopy(model).to('cpu')
    bestEpoch = epochStart

    bestModelData = {'model': bestModel, 'epoch': bestEpoch,
                     'validationLoss': prevBestValidationLoss}

    for epochID in range(numOfEpochs):

        print(
            f"\n\t\t\tTRAINING EPOCH: {epochID + epochStart} of {epochStart + numOfEpochs-1}\n-------------------------------")
        # Do training over all batches
        trainLossHistory[epochID], numOfUpdatesForEpoch = TrainModel(trainingDataset, model, lossFcn, optimizer, epochID,
                                                                     device, taskType, lr_scheduler)
        numOfUpdates += numOfUpdatesForEpoch
        print('Current total number of updates: ', numOfUpdates)

        # Do validation over all batches
        validationLossHistory[epochID], validationData = ValidateModel(
            validationDataset, model, lossFcn, device, taskType)

        # If validation loss is better than previous best, update best model
        if validationLossHistory[epochID] < prevBestValidationLoss:
            # Replace best model with current model
            bestModel = copy.deepcopy(model).to('cpu')
            bestEpoch = epochID + epochStart
            prevBestValidationLoss = validationLossHistory[epochID]

            # Overwrite dictionary with best model data
            bestModelData['model'] = bestModel
            bestModelData['epoch'] = bestEpoch
            bestModelData['validationLoss'] = prevBestValidationLoss

        print(
            f"\n\nCurrent best model found at epoch: {bestEpoch} with validation loss: {prevBestValidationLoss}")

        # Update Tensorboard if enabled
        # if enableTensorBoard:
        # tensorBoardWriter.add_scalar(lossLogName + "/train", trainLossHistory[epochID], epochID + epochStart)
        # tensorBoardWriter.add_scalar(lossLogName + "/validation", validationLossHistory[epochID], epochID + epochStart)
        # entriesTagDict = {'Training': trainLossHistory[epochID], 'Validation': validationLossHistory[epochID]}
        # tensorBoardWriter.add_scalars(lossLogName, entriesTagDict, epochID)

        mlflow.log_metric('Training loss - ' + lossLogName,
                          trainLossHistory[epochID], step=epochID + epochStart)
        mlflow.log_metric('Validation loss - ' + lossLogName,
                          validationLossHistory[epochID], step=epochID + epochStart)

        if 'WorstLossAcrossBatches' in validationData.keys():
            mlflow.log_metric('Validation Worst Loss across batches',
                              validationData['WorstLossAcrossBatches'], step=epochID + epochStart)

        if enableSave:
            # NOTE: models are all saved as traced models
            if not (os.path.isdir(checkpointDir)):
                os.mkdir(checkpointDir)

            raise NotImplementedError('Update required. Function call not found')
            exampleInput = GetSamplesFromDataset(validationDataset, 1)[0][0].reshape(
                1, -1)  # Get single input sample for model saving
            
            modelSaveName = os.path.join(
                checkpointDir, modelName + '_' + AddZerosPadding(epochID + epochStart, stringLength=4))
            
            SaveModel(model, modelSaveName, example_input=exampleInput, target_device=device, save_mode='traced_dynamo')

        # Optuna functionalities
        # Report validation loss to Optuna pruner
        trial.report(validationLossHistory[epochID], step=epochID)
        if trial.should_prune():
            # End mlflow run and raise exception to prune trial
            mlflow.end_run(status='KILLED')
            raise optuna.TrialPruned()

    mlflow.end_run(status='FINISHED')

    # tensorBoardWriter.flush() # Force tensorboard to write data to disk
    # Print best model and epoch
    return bestModelData
