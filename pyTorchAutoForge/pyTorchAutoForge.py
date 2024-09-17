'''
Module collecting utilities function building upon PyTorch to speed up prototyping, training and testing of Neural Nets.
Utilities for monitoring training, validation and testing of models, as well as saving and loading models and datasets are included.
The module also includes functions to export and load models to/from ONNx format, as well as a MATLAB wrapper class for model evaluation.
NOTE: This however, requires TorchTCP module (not included yet) and a MATLAB tcpclient based interface.
Created by PeterC - 04-05-2024. Current version: v0.1 (30-06-2024)
'''


# Import modules
from typing import Optional, Any, Union
import torch, mlflow, optuna
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
from dataclasses import dataclass, asdict

# import datetime
import numpy as np
import sys, os, signal, copy, inspect, subprocess
import psutil
import onnx
from onnx import version_converter
from typing import Union

from torch.utils.tensorboard import SummaryWriter # Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim
import torch.nn.functional as F # Module to apply activation functions in forward pass instead of defining them in the model class

# class FDNNbuilder:
#     def __init__():

# %% Function to get device if not passed to trainModel and validateModel()
def GetDevice():
    device = ("cuda:0"
              if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu" )
    #print(f"Using {device} device")
    return device

# %% Function to perform one step of training of a model using dataset and specified loss function - 04-05-2024
# Updated by PC 04-06-2024

def TrainModel(dataloader:DataLoader, model:nn.Module, lossFcn:nn.Module, 
               optimizer, epochID:int, device=GetDevice(), taskType:str='classification', lr_scheduler=None,
               swa_scheduler=None, swa_model=None, swa_start_epoch:int=15) -> Union[float, int]:

    model.train() # Set model instance in training mode ("informing" backend that the training is going to start)

    counterForPrint = np.round(len(dataloader)/75)
    numOfUpdates = 0

    if swa_scheduler is not None or lr_scheduler is not None:
        mlflow.log_metric('Learning rate', optimizer.param_groups[0]['lr'], step=epochID)

    print('Starting training loop using learning rate: {:.11f}'.format(optimizer.param_groups[0]['lr']))

    for batchCounter, (X, Y) in enumerate(dataloader): # Recall that enumerate gives directly both ID and value in iterable object

        # Get input and labels and move to target device memory
        X, Y = X.to(device), Y.to(device) # Define input, label pairs for target device

        # Perform FORWARD PASS to get predictions
        predVal = model(X) # Evaluate model at input
        trainLossOut = lossFcn(predVal, Y) # Evaluate loss function to get loss value (this returns loss function instance, not a value)

        if isinstance(trainLossOut, dict):
            trainLoss = trainLossOut.get('lossValue')
            keys = [key for key in trainLossOut.keys() if key != 'lossValue']
            # Log metrics to MLFlow after converting dictionary entries to float
            mlflow.log_metrics(  {key: value.item() if isinstance(value, torch.Tensor) else value for key, value in trainLossOut.items()}, step=numOfUpdates)
                
        else:
            trainLoss = trainLossOut
            keys = []

        # Perform BACKWARD PASS to update parameters
        trainLoss.backward()  # Compute gradients
        optimizer.step()      # Apply gradients from the loss
        optimizer.zero_grad() # Reset gradients for next iteration

        numOfUpdates += 1

        if batchCounter % counterForPrint == 0: # Print loss value 
            trainLoss, currentStep = trainLoss.item(), (batchCounter + 1) * len(X)
            #print(f"Training loss value: {trainLoss:>7f}  [{currentStep:>5d}/{size:>5d}]")
            #if keys != []:
            #    print("\t",", ".join([f"{key}: {trainLossOut[key]:.4f}" for key in keys]))    # Update learning rate if scheduler is provided
    # Perform step of SWA if enabled
    if swa_model is not None and epochID >= swa_start_epoch:
        # Update SWA model parameters
        swa_model.train()
        swa_model.update_parameters(model)
        # Update SWA scheduler
        #swa_scheduler.step()
        torch.optim.swa_utils.update_bn(dataloader, swa_model, device=device) # Update batch normalization layers for swa model
    #else:
    if lr_scheduler is not None:
        lr_scheduler.step()
        print('\n')
        print('Learning rate modified to: ', lr_scheduler.get_last_lr())
        print('\n')

    return trainLoss, numOfUpdates
    
# %% Function to validate model using dataset and specified loss function - 04-05-2024
# Updated by PC 04-06-2024

def ValidateModel(dataloader:DataLoader, model:nn.Module, lossFcn:nn.Module, device=GetDevice(), taskType:str='classification') -> Union[float, dict]:
    
    # Get size of dataset (How many samples are in the dataset)
    size = len(dataloader.dataset) 

    model.eval() # Set the model in evaluation mode
    validationLoss = 0 # Accumulation variables
    batchMaxLoss = 0

    validationData =  {} # Dictionary to store validation data

    # Initialize variables based on task type
    if taskType.lower() == 'classification': 
        correctOuputs = 0

    elif taskType.lower() == 'regression':
        avgRelAccuracy = 0.0
        avgAbsAccuracy = 0.0

    elif taskType.lower() == 'custom':
        print('TODO')

    with torch.no_grad(): # Tell torch that gradients are not required
        # TODO: try to modify function to perform one single forward pass for the entire dataset

        # Backup the original batch size
        original_dataloader =  dataloader
        original_batch_size = dataloader.batch_size
        
        # Temporarily initialize a new dataloader for validation
        allocMem = torch.cuda.memory_allocated(0)
        freeMem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        estimated_memory_per_sample = allocMem / original_batch_size
        newBathSizeTmp = min(round(0.5 * freeMem / estimated_memory_per_sample), 2048)

        dataloader = DataLoader(dataloader.dataset, batch_size=newBathSizeTmp, shuffle=False, drop_last=False)
        lossTerms = {}
        numberOfBatches = len(dataloader)

        for X,Y in dataloader:
            # Get input and labels and move to target device memory
            X, Y = X.to(device), Y.to(device)  

            # Perform FORWARD PASS
            predVal = model(X) # Evaluate model at input
            tmpLossVal = lossFcn(predVal, Y) # Evaluate loss function and accumulate

            if isinstance(tmpLossVal, dict):
                tmpVal = tmpLossVal.get('lossValue').item()

                validationLoss += tmpVal
                if batchMaxLoss < tmpVal:
                    batchMaxLoss = tmpVal

                keys = [key for key in tmpLossVal.keys() if key != 'lossValue']
                # Sum all loss terms for each batch if present in dictionary output
                for key in keys:
                    lossTerms[key] = lossTerms.get(key, 0) + tmpLossVal[key].item()
            else:

                validationLoss += tmpLossVal.item()
                if batchMaxLoss < tmpLossVal.item():
                    batchMaxLoss = tmpLossVal.item()

            validationData['WorstLossAcrossBatches'] = batchMaxLoss

            if taskType.lower() == 'classification': 
                # Determine if prediction is correct and accumulate
                # Explanation: get largest output logit (the predicted class) and compare to Y. 
                # Then convert to float and sum over the batch axis, which is not necessary if size is single prediction
                correctOuputs += (predVal.argmax(1) == Y).type(torch.float).sum().item() 

            #elif taskType.lower() == 'regression':
            #    #print('TODO')
            #elif taskType.lower() == 'custom':
            #    print('TODO')
            #predVal = model(dataX) # Evaluate model at input
            #validationLoss += lossFcn(predVal, dataY).item() # Evaluate loss function and accumulate
    
    # Log additional metrics to MLFlow if any
    if lossTerms != {}:
        lossTerms = {key: (value / numberOfBatches) for key, value in lossTerms.items()}
        mlflow.log_metrics(lossTerms, step=0)

    # Restore the original batch size
    dataloader = original_dataloader

    # EXPERIMENTAL: Try to perform one single forward pass for the entire dataset (MEMORY BOUND)
    #with torch.no_grad():
    #    TENSOR_VALIDATION_EVAL = False
    #    if TENSOR_VALIDATION_EVAL:
    #        dataX = []
    #        dataY = []
    #    # NOTE: the memory issue is in transforming the list into a torch tensor on the GPU. For some reasons
    #    # the tensor would require 81 GBits of memory.
    #        for X, Y in dataloader:
    #            dataX.append(X)
    #            dataY.append(Y)
    #        # Concatenate all data in a single tensor
    #        dataX = torch.cat(dataX, dim=0).to(device)
    #        dataY = torch.cat(dataY, dim=0).to(device)
    #        predVal_dataset = model(dataX) # Evaluate model at input
    #        validationLoss_dataset = lossFcn(predVal_dataset, dataY).item() # Evaluate loss function and accumulate


    if taskType.lower() == 'classification': 
        validationLoss /= numberOfBatches # Compute batch size normalized loss value
        correctOuputs /= size # Compute percentage of correct classifications over batch size
        print(f"\n VALIDATION ((Classification) Accuracy: {(100*correctOuputs):>0.2f}%, Avg loss: {validationLoss:>8f} \n")

    elif taskType.lower() == 'regression':
        #print('TODO')
        correctOuputs = None

        if isinstance(tmpLossVal, dict):
            keys = [key for key in tmpLossVal.keys() if key != 'lossValue']
        
        validationLoss /= numberOfBatches
        print(f"\n VALIDATION (Regression) Avg loss: {validationLoss:>0.5f}, Max batch loss: {batchMaxLoss:>0.5f}\n")
        #print(f"Validation (Regression): \n Avg absolute accuracy: {avgAbsAccuracy:>0.1f}, Avg relative accuracy: {(100*avgRelAccuracy):>0.1f}%, Avg loss: {validationLoss:>8f} \n")

    elif taskType.lower() == 'custom':
        print('TODO')

    return validationLoss, validationData
    # TODO: add command for Tensorboard here


# %% Class to define a custom loss function for training, validation and testing - 01-06-2024
# NOTE: Function EvalLossFcn must be implemented using Torch operations to work!

class CustomLossFcn(nn.Module):
    '''Custom loss function based class, instantiated by specifiying a loss function (callable object) and optionally, a dictionary containing parameters required for the evaluation'''
    
    def __init__(self, EvalLossFcn:callable, lossParams:dict = None) -> None:
        '''Constructor for CustomLossFcn class'''
        super(CustomLossFcn, self).__init__() # Call constructor of nn.Module
        
        if len((inspect.signature(EvalLossFcn)).parameters) >= 2:
            self.LossFcnObj = EvalLossFcn
        else: 
            raise ValueError('Custom EvalLossFcn must take at least two inputs: inputVector, labelVector')    

        # Store loss function parameters dictionary
        self.lossParams = lossParams 

    #def setTrainingMode(self):
    #    self.lossParams = lossParams 
        
    #def setEvalMode(self):

    def forward(self, predictVector, labelVector):
        ''''Forward pass method to evaluate loss function on input and label vectors using EvalLossFcn'''
        lossBatch = self.LossFcnObj(predictVector, labelVector, self.lossParams)

        if isinstance(lossBatch, torch.Tensor):
            assert(lossBatch.dim() == 0)
        elif isinstance(lossBatch, dict):
            assert(lossBatch.get('lossValue').dim() == 0)
        else:
            raise ValueError('EvalLossFcn must return a scalar loss value (torch tensor) or a dictionary with a "lossValue" key')
        
        return lossBatch
   

# %% Function to save model state - 04-05-2024, updated 11-06-2024

# %% Generic Dataset class for Supervised learning - 30-05-2024
# Base class for Supervised learning datasets
# Reference for implementation of virtual methods: https://stackoverflow.com/questions/4714136/how-to-implement-virtual-methods-in-python

from abc import abstractmethod
from abc import ABCMeta


class GenericSupervisedDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, inputDataPath:str='inputData/', labelsDataPath:str='labelsData/', 
                 datasetType:str='train', transform=None, target_transform=None):
        # Store input and labels sources
        self.labelsDir = labelsDataPath
        self.inputDir = inputDataPath

        # Initialize transform objects
        self.transform = transform
        self.target_transform = target_transform

        # Set the dataset type (train, test, validation)
        self.datasetType = datasetType

    def __len__(self):
        return len() # TODO

    @abstractmethod
    def __getLabelsData__(self):
        raise NotImplementedError()
        # Get and store labels vector
        self.labels # TODO: "Read file" of some kind goes here. Best current option: write to JSON 

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()
        return inputVec, label


# %% Function to get model checkpoint and load it into nn.Module for training restart - 09-06-2024
def LoadModelAtCheckpoint(model:nn.Module, modelSavePath:str='./checkpoints', modelName:str='trainedModel', modelEpoch:int=0) -> nn.Module: 
    # TODO: add checks that model and checkpoint matches: how to? Check number of parameters?

    # Create path to model state file
    checkPointPath = os.path.join(modelSavePath, modelName + '_' + AddZerosPadding(modelEpoch, stringLength=4))

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
        raise ValueError('Specified model state file not found. Check input path.') 
    
    # Get last saving of model (NOTE: getmtime does not work properly. Use scandir + list comprehension)
    #with os.scandir(modelSavePath) as it:
        #modelNamesWithTime = [(entry.name, entry.stat().st_mtime) for entry in it if entry.is_file()]
        #modelName = sorted(modelNamesWithTime, key=lambda x: x[1])[-1][0]

    return loadedModel

# %%  Data loader indexer class - PeterC - 23-07-2024
class dataloaderIndex:
    '''
    Class to index dataloaders for training and validation datasets. Performs splitting if validation loader is not provided.
    Created by PeterC, 23-07-2024
    '''
    def __init__(self, trainLoader:DataLoader, validLoader:Optional[DataLoader] = None) -> None:
        if not(isinstance(trainLoader, DataLoader)):
            raise TypeError('Training dataloader is not of type "DataLoader"!')

        if validLoader is None:
            # Perform random splitting of training data to get validation dataset
            print('No validation dataset provided: training dataset automatically split with ratio 80/20')
            trainingSize = int(0.8 * len(trainLoader.dataset))
            validationSize = len(trainLoader.dataset) - trainingSize

            # Split the dataset
            trainingData, validationData = torch.utils.data.random_split(trainLoader.dataset, [trainingSize, validationSize])

            # Create dataloaders
            self.TrainingDataLoader = DataLoader(trainingData, batch_size=trainLoader.batch_size, shuffle=True, 
                                                 num_workers=trainLoader.num_workers, drop_last=trainLoader.drop_last)
            self.ValidationDataLoader = DataLoader(validationData, batch_size=trainLoader.batch_size, shuffle=True,
                                                   num_workers=trainLoader.num_workers, drop_last=False)
        else:

            self.TrainingDataLoader = trainLoader

            if not(isinstance(validLoader, DataLoader)):
                raise TypeError('Validation dataloader is not of type "DataLoader"!')
            
            self.ValidationDataLoader = validLoader

    def getTrainloader(self) -> DataLoader:
        return self.TrainingDataLoader
    
    def getValidationLoader(self) -> DataLoader:
        return self.ValidationDataLoader

# %% TRAINING and VALIDATION template function - 04-06-2024
def TrainAndValidateModel(dataloaderIndex:dataloaderIndex, model:nn.Module, lossFcn: nn.Module, optimizer, options:dict={}):
    # NOTE: is the default dictionary considered as "single" object or does python perform a merge of the fields?

    # TODO: For merging of options: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
    #if options is None:
    #    options = {}
    #
    ## Merge user-provided options with default options
    # combined_options = {**default_options, **options}
    # Now use combined_options in the function
    # taskType = combined_options['taskType']
    # device = combined_options['device']
    # epochs = combined_options['epochs']
    # tensorboard = combined_options['Tensorboard']
    # save_checkpoints = combined_options['saveCheckpoints']
    # checkpoints_out_dir = combined_options['checkpointsOutDir']
    # model_name = combined_options['modelName']
    # load_checkpoint = combined_options['loadCheckpoint']
    # loss_log_name = combined_options['lossLogName']
    # epoch_start = combined_options['epochStart']

    # Setup options from input dictionary
    taskType            = options.get('taskType', 'regression') # NOTE: Classification is not well developed (July, 2024)
    device              = options.get('device', GetDevice())
    numOfEpochs         = options.get('epochs', 10)
    enableSave          = options.get('saveCheckpoints', True)
    checkpointDir       = options.get('checkpointsOutDir', './checkpoints')
    modelName           = options.get('modelName', 'trainedModel')
    lossLogName         = options.get('lossLogName', 'Loss_value') 
    epochStart          = options.get('epochStart', 0)

    swa_scheduler       = options.get('swa_scheduler', None)    
    swa_model           = options.get('swa_model', None)
    swa_start_epoch     = options.get('swa_start_epoch', 15)

    child_run = None
    child_run_name = None 
    parent_run = mlflow.active_run()
    parent_run_name = parent_run.info.run_name
    
    lr_scheduler       = options.get('lr_scheduler', None)
    # Default early stopping for regression: "minimize" direction
    #early_stopper = options.get('early_stopper', early_stopping=EarlyStopping(monitor="lossValue", patience=5, verbose=True, mode="min"))
    early_stopper = options.get('early_stopper', None)

    # Get Torch dataloaders
    trainingDataset   = dataloaderIndex.getTrainloader()
    validationDataset = dataloaderIndex.getValidationLoader()

    # Configure Tensorboard
    #if 'logDirectory' in options.keys():
    #    logDirectory = options['logDirectory']
    #else:
    #    currentTime = datetime.datetime.now()
    #    formattedTimestamp = currentTime.strftime('%d-%m-%Y_%H-%M') # Format time stamp as day, month, year, hour and minute
    #    logDirectory = './tensorboardLog_' + modelName + formattedTimestamp
        
    #if not(os.path.isdir(logDirectory)):
    #    os.mkdir(logDirectory)
    #tensorBoardWriter = ConfigTensorboardSession(logDirectory, portNum=tensorBoardPortNum)

    # If training is being restarted, attempt to load model
    if options['loadCheckpoint'] == True:
        raise NotImplementedError('Training restart from checkpoint REMOVED. Not updated with mlflow yet.')
        model = LoadModelAtCheckpoint(model, options['checkpointsInDir'], modelName, epochStart)

    # Move model to device if possible (check memory)
    try:
        print('Moving model to selected device:', device)
        model = model.to(device) # Create instance of model using device 
    except Exception as exception:
        # Add check on error and error handling if memory insufficient for training on GPU:
        print('Attempt to load model in', device, 'failed due to error: ', repr(exception))

    #input('\n-------- PRESS ENTER TO START TRAINING LOOP --------\n')
    trainLossHistory = np.zeros(numOfEpochs)
    validationLossHistory = np.zeros(numOfEpochs)

    numOfUpdates = 0
    bestValidationLoss = 1E10
    bestSWAvalidationLoss = 1E10

    bestModel = copy.deepcopy(model).to('cpu')  # Deep copy the initial state of the model and move it to the CPU
    bestEpoch = epochStart

    if swa_model != None:
        bestSWAmodel = copy.deepcopy(model).to('cpu')

    # TRAINING and VALIDATION LOOP
    for epochID in range(numOfEpochs):

        print(f"\n\t\t\tTRAINING EPOCH: {epochID + epochStart} of {epochStart + numOfEpochs-1}\n-------------------------------")
        # Do training over all batches
        trainLossHistory[epochID], numOfUpdatesForEpoch = TrainModel(trainingDataset, model, lossFcn, optimizer, epochID, device, 
                                                                     taskType, lr_scheduler, swa_scheduler, swa_model, swa_start_epoch)
        numOfUpdates += numOfUpdatesForEpoch
        print('Current total number of updates: ', numOfUpdates)

        # Do validation over all batches
        validationLossHistory[epochID], validationData = ValidateModel(validationDataset, model, lossFcn, device, taskType) 

        # If validation loss is better than previous best, update best model
        if validationLossHistory[epochID] < bestValidationLoss:
            bestModel = copy.deepcopy(model).to('cpu') # Replace best model with current model 
            bestEpoch = epochID + epochStart
            bestValidationLoss = validationLossHistory[epochID]

            bestModelData = {'model': bestModel, 'epoch': bestEpoch,
                             'validationLoss': bestValidationLoss}
            
        print(f"Current best model found at epoch: {bestEpoch} with validation loss: {bestValidationLoss}")

        # SWA handling: if enabled, evaluate validation loss of SWA model, then decide if to update or reset
        if swa_model != None and epochID >= swa_start_epoch:
            
            # Verify swa_model on the validation dataset
            swa_model.eval()
            swa_validationLoss, _ = ValidateModel(validationDataset, swa_model, lossFcn, device, taskType)
            swa_model.train()
            print(f"Current SWA model found at epoch: {epochID} with validation loss: {swa_validationLoss}")

            #if swa_validationLoss < bestSWAvalidationLoss:
                # Update best SWA model
            bestSWAvalidationLoss = swa_validationLoss
            bestSWAmodel = copy.deepcopy(swa_model).to('cpu')
            swa_has_improved = True
            #else: 
            #    # Reset to previous best model
            #    swa_model = copy.deepcopy(bestSWAmodel).to(device)
            #    swa_has_improved = False

            # Log data to mlflow by opening children run 

            if child_run_name is None and child_run is None:
                child_run_name = parent_run_name + '-SWA'
                child_run = mlflow.start_run(run_name=child_run_name, nested=True)
            mlflow.log_metric('SWA Best validation loss', bestSWAvalidationLoss, step=epochID + epochStart, run_id=child_run.info.run_id)
        else:
            swa_has_improved = False

        # Re-open parent run scope
        mlflow.start_run(run_id=parent_run.info.run_id, nested=True)

        # Log parent run data
        mlflow.log_metric('Training loss - '+ lossLogName, trainLossHistory[epochID], step=epochID + epochStart)
        mlflow.log_metric('Validation loss - '+ lossLogName, validationLossHistory[epochID], step=epochID + epochStart)

        if 'WorstLossAcrossBatches' in validationData.keys():
            mlflow.log_metric('Validation Worst Loss across batches', validationData['WorstLossAcrossBatches'], step=epochID + epochStart)

        if enableSave:
            if not(os.path.isdir(checkpointDir)):
                os.mkdir(checkpointDir)

            exampleInput = GetSamplesFromDataset(validationDataset, 1)[0][0].reshape(1, -1) # Get single input sample for model saving
            modelSaveName = os.path.join(checkpointDir, modelName + '_' + AddZerosPadding(epochID + epochStart, stringLength=4))
            SaveTorchModel(model, modelSaveName, saveAsTraced=True, exampleInput=exampleInput, targetDevice=device)

            if swa_model != None and swa_has_improved:
                swa_model.eval()
                SaveTorchModel(swa_model, modelSaveName + '_SWA', saveAsTraced=True, exampleInput=exampleInput, targetDevice=device)
                swa_model.train()

        # %% MODEL PREDICTION EXAMPLES
        examplePrediction, exampleLosses, inputSampleTensor, labelsSampleTensor = EvaluateModel(validationDataset, model, lossFcn, device, 20)

        if swa_model is not None and epochID >= swa_start_epoch:
            # Test prediction of SWA model on the same input samples
            swa_model.eval()
            swa_examplePrediction, swa_exampleLosses, _, _ = EvaluateModel(
                    validationDataset, swa_model, lossFcn, device, 20, inputSampleTensor, labelsSampleTensor)
            swa_model.train()

        #mlflow.log_artifacts('Prediction samples: ', validationLossHistory[epochID])
        
        #mlflow.log_param(f'ExamplePredictionList', list(examplePrediction))
        #mlflow.log_param(f'ExampleLosses', list(exampleLosses))

        print('\n  Random Sample predictions from validation dataset:\n')

        for id in range(examplePrediction.shape[0]):

            formatted_predictions = ['{:.5f}'.format(num) for num in examplePrediction[id, :]]
            formatted_loss = '{:.5f}'.format(exampleLosses[id])
            print(f'\tPrediction: {formatted_predictions} --> Loss: {formatted_loss}')

        print('\t\t Average prediction loss: {avgPred}\n'.format(avgPred=torch.mean(exampleLosses)) )

        if swa_model != None and epochID >= swa_start_epoch:
            for id in range(examplePrediction.shape[0]):
                    formatted_predictions = ['{:.5f}'.format(num) for num in swa_examplePrediction[id, :]]
                    formatted_loss = '{:.5f}'.format(swa_exampleLosses[id])
                    print(f'\tSWA Prediction: {formatted_predictions} --> Loss: {formatted_loss}')

            print('\t\t SWA Average prediction loss: {avgPred}\n'.format(
                avgPred=torch.mean(swa_exampleLosses)))

        # Perform step of Early stopping if enabled
        if early_stopper is not None:
            print('Early stopping NOT IMPLEMENTED. Skipping...')
            #early_stopper.step(validationLossHistory[epochID])
            #if early_stopper.early_stop:
            #    mlflow.end_run(status='KILLED')
            #    print('Early stopping triggered at epoch: {epochID}'.format(epochID=epochID))
            #    break
            #earlyStopping(validationLossHistory[epochID], model, bestModelData, options)
    if swa_model != None and epochID >= swa_start_epoch:
        # End nested child run
        mlflow.end_run(status='FINISHED')
    # End parent run
    mlflow.end_run(status='FINISHED')

    if swa_model is not None:
        bestModelData['swa_model'] = bestSWAmodel

    return bestModelData, trainLossHistory, validationLossHistory, inputSampleTensor

# %% New version of TrainAndValidateModel with MLFlow logging specifically design for Optuna studies - 11-07-2024
def TrainAndValidateModelForOptunaOptim(trial, dataloaderIndex: dict, model: nn.Module, lossFcn: nn.Module, optimizer, options: dict = {}):

    # NOTE: Classification is not well developed (July, 2024). Default is regression
    taskType = options.get('taskType', 'regression')
    device = options.get('device', GetDevice())
    numOfEpochs = options.get('epochs', 10)
    enableSave = options.get('saveCheckpoints', True)
    checkpointDir = options.get('checkpointsOutDir', './checkpoints')
    modelName = options.get('modelName', 'trainedModel')
    lossLogName = options.get('lossLogName', 'Loss_value')
    epochStart = options.get('epochStart', 0)

    lr_scheduler = options.get('lr_scheduler', None)


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
        model = LoadModelAtCheckpoint(model, options['checkpointsInDir'], modelName, epochStart)

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

        print(f"\n\t\t\tTRAINING EPOCH: {epochID + epochStart} of {epochStart + numOfEpochs-1}\n-------------------------------")
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
                
        print(f"\n\nCurrent best model found at epoch: {bestEpoch} with validation loss: {prevBestValidationLoss}")

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
            mlflow.log_metric('Validation Worst Loss across batches', validationData['WorstLossAcrossBatches'], step=epochID + epochStart)

        if enableSave:
            # NOTE: models are all saved as traced models
            if not (os.path.isdir(checkpointDir)):
                os.mkdir(checkpointDir)

            exampleInput = GetSamplesFromDataset(validationDataset, 1)[0][0].reshape(
                1, -1)  # Get single input sample for model saving
            modelSaveName = os.path.join(
                checkpointDir, modelName + '_' + AddZerosPadding(epochID + epochStart, stringLength=4))
            SaveTorchModel(model, modelSaveName, saveAsTraced=True,
                           exampleInput=exampleInput, targetDevice=device)

        # Optuna functionalities
        trial.report(validationLossHistory[epochID], step=epochID) # Report validation loss to Optuna pruner
        if trial.should_prune():
            # End mlflow run and raise exception to prune trial
            mlflow.end_run(status='KILLED')
            raise optuna.TrialPruned()

    mlflow.end_run(status='FINISHED')

    # tensorBoardWriter.flush() # Force tensorboard to write data to disk
    # Print best model and epoch
    return bestModelData



# %% Model evaluation function on a random number of samples from dataset - 06-06-2024
# Possible way to solve the issue of having different cost function terms for training and validation --> add setTrain and setEval methods to switch between the two

def EvaluateModel(dataloader:DataLoader, model:nn.Module, lossFcn: nn.Module, device=GetDevice(), numOfSamples:int=10, 
                  inputSample:torch.tensor=None, labelsSample:torch.tensor=None) -> np.array:
    '''Torch model evaluation function to perform inference using either specified input samples or input dataloader'''
    model.eval() # Set model in prediction mode
    with torch.no_grad(): 
        if inputSample is None and labelsSample is None:
            # Get some random samples from dataloader as list
            extractedSamples = GetSamplesFromDataset(dataloader, numOfSamples)

            # Create input array as torch tensor
            X = torch.zeros(len(extractedSamples), extractedSamples[0][0].shape[0])
            Y = torch.zeros(len(extractedSamples), extractedSamples[0][1].shape[0])

            #inputSampleList = []
            for id, (inputVal, labelVal) in enumerate(extractedSamples):
                X[id, :] = inputVal
                Y[id, :] = labelVal

            #inputSampleList.append(inputVal.reshape(1, -1))

            # Perform FORWARD PASS
            examplePredictions = model(X.to(device)) # Evaluate model at input

            # Compute loss for each input separately
            exampleLosses = torch.zeros(examplePredictions.size(0))

            examplePredictionList = []
            for id in range(examplePredictions.size(0)):

                # Get prediction and label samples 
                examplePredictionList.append(examplePredictions[id, :].reshape(1, -1))
                labelSample = Y[id,:].reshape(1, -1)

                # Evaluate loss function
                outLossVar = lossFcn(examplePredictionList[id].to(device), labelSample.to(device))

                if isinstance(outLossVar, dict):
                    exampleLosses[id] = outLossVar.get('lossValue').item()
                else:
                    exampleLosses[id] = outLossVar.item()
                    
        elif inputSample is not None and labelsSample is not None:
            # Perform FORWARD PASS # NOTE: NOT TESTED
            X = inputSample
            Y = labelsSample

            examplePredictions = model(X.to(device)) # Evaluate model at input

            exampleLosses = torch.zeros(examplePredictions.size(0))
            examplePredictionList = []

            for id in range(examplePredictions.size(0)):

                # Get prediction and label samples 
                examplePredictionList.append(examplePredictions[id, :].reshape(1, -1))
                labelSample = Y[id,:].reshape(1, -1)

                # Evaluate loss function
                outLossVar = lossFcn(examplePredictionList[id].to(device), labelSample.to(device))

                if isinstance(outLossVar, dict):
                    exampleLosses[id] = outLossVar.get('lossValue').item()
                else:
                    exampleLosses[id] = outLossVar.item()
        else:
            raise ValueError('Either both inputSample and labelsSample must be provided or neither!')
        
        return examplePredictions, exampleLosses, X.to(device), Y.to(device)

                
# %% Function to extract specified number of samples from dataloader - 06-06-2024
# ACHTUNG: TO REWORK USING NEXT AND ITER!
def GetSamplesFromDataset(dataloader: DataLoader, numOfSamples:int=10):

    samples = []
    for batch in dataloader:
        for sample in zip(*batch): # Construct tuple (X,Y) from batch
            samples.append(sample)

            if len(samples) == numOfSamples:
                return samples
                   
    return samples

# %% Model Graph visualization function based on Netron module # TODO


#inputImageSize:list, kernelSizes:list, OutputChannelsSizes:list, PoolingLayersSizes:list, inputChannelSize:int=1, withBiases=True

# Auxiliar functions
def ComputeConv2dOutputSize(inputSize:Union[list, np.array, torch.tensor], kernelSize=3, strideSize=1, paddingSize=0):
    '''Compute output size and number of features maps (channels, i.e. volume) of a 2D convolutional layer. 
       Input size must be a list, numpy array or a torch tensor with 2 elements: [height, width].'''
    return int((inputSize[0] + 2*paddingSize - (kernelSize-1)-1) / strideSize + 1), int((inputSize[1] + 2*paddingSize - (kernelSize-1)-1) / strideSize + 1)

def ComputePooling2dOutputSize(inputSize:Union[list, np.array, torch.tensor], kernelSize=2, strideSize=2, paddingSize=0):
    '''Compute output size and number of features maps (channels, i.e. volume) of a 2D max/avg pooling layer. 
       Input size must be a list, numpy array or a torch tensor with 2 elements: [height, width].'''
    return int(( (inputSize[0] + 2*paddingSize - (kernelSize-1)-1) / strideSize) + 1), int(( (inputSize[1] + 2*paddingSize - (kernelSize-1)-1) / strideSize) + 1)

# ConvBlock 2D and flatten sizes computation (SINGLE BLOCK)
def ComputeConvBlockOutputSize(inputSize:Union[list, np.array, torch.tensor], outChannelsSize:int, 
                               convKernelSize:int=3, poolingkernelSize:int=2, 
                               convStrideSize:int=1, poolingStrideSize:int=None, 
                               convPaddingSize:int=0, poolingPaddingSize:int=0):
    
    # TODO: modify interface to use something like a dictionary with the parameters, to make it more fexible and avoid the need to pass all the parameters

    '''Compute output size and number of features maps (channels, i.e. volume) of a ConvBlock layer. 
       Input size must be a list, numpy array or a torch tensor with 2 elements: [height, width].'''

    if poolingStrideSize is None:
        poolingStrideSize = poolingkernelSize
        
    # Compute output size of Conv2d and Pooling2d layers
    conv2dOutputSize = ComputeConv2dOutputSize(inputSize, convKernelSize, convStrideSize, convPaddingSize)
    
    if conv2dOutputSize[0] < poolingkernelSize or conv2dOutputSize[1] < poolingkernelSize:
        raise ValueError('Pooling kernel size is larger than output size of Conv2d layer. Check configuration.')
    
    convBlockOutputSize = ComputePooling2dOutputSize(conv2dOutputSize, poolingkernelSize, poolingStrideSize, poolingPaddingSize)

    # Compute total number of features after ConvBlock as required for the fully connected layers
    conv2dFlattenOutputSize = convBlockOutputSize[0] * convBlockOutputSize[1] * outChannelsSize

    return convBlockOutputSize, conv2dFlattenOutputSize


# %% MATLAB wrapper class for Torch models evaluation - 11-06-2024
class TorchModel_MATLABwrap():
    def __init__(self, trainedModelName:str, trainedModelPath:str) -> None:
        # Get available device
        self.device = GetDevice()

        # Load model state and state
        trainedModel = LoadTorchModel(None, trainedModelName, trainedModelPath, loadAsTraced=True)
        trainedModel.eval() # Set model in evaluation mode
        self.trainedModel = trainedModel.to(self.device)


    def forward(self, inputSample:np.array, numBatches:int=1):
        '''Forward method to perform inference for ONE sample input using trainedModel'''
        if inputSample.dtype is not np.float32:
            inputSample = np.float32(inputSample)

        print('Performing formatting of input array to pass to model...')

        # TODO: check the input is exactly identical to what the model receives using EvaluateModel() loading from dataset!        
        # Convert numpy array into torch.tensor for model inference
        X = torch.tensor(inputSample).reshape(numBatches, -1)
                        
        # ########### DEBUG ######################: 
        print('Evaluating model using batch input: ', X)
        ############################################

        # Perform inference using model
        Y = self.trainedModel(X.to(self.device))

        # ########### DEBUG ######################: 
        print('Model prediction: ', Y)
        ############################################

        return Y.detach().cpu().numpy() # Move to cpu and convert to numpy
    



# TODO
# %% EXPERIMENTAL: Network AutoBuilder class - 02-07-2024 (WIP)
class ModelAutoBuilder():
    #print('TODO')
    def __init__(self):
        pass


# %% EXPERIMENTAL: fully autonomous ModelAutoDesigner class
class ModelAutoDesigner():
    #print('TODO')
    def __init__(self):
        pass

# %% MAIN 
def main():
    print('In this script, main does actually nothing lol ^_^.')
    
if __name__== '__main__':
    main()