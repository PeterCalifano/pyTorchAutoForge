

# TODO Add yaml interface for training, compatible with mlflow and optuna
# The idea is to let the user specify all the parameters in a yaml file, which is then loaded and used
# to set the configuration class. Default values are specified as the class defaults. 
# Loading methods only modify the parameters the user has specified


from typing import Optional, Any, Union, IO
import torch, mlflow, os
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict, fields, Field, MISSING

from pyTorchAutoForge.datasets import DataloaderIndex
from pyTorchAutoForge.utils.utils import GetDevice, AddZerosPadding, GetSamplesFromDataset
from pyTorchAutoForge.api.torch import SaveTorchModel

# import datetime
import yaml
import copy

# Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim


# %% Training and validation manager class - 22-06-2024 (WIP)
# TODO: Features to include:
# 1) Multi-process/multi-threading support for training and validation of multiple models in parallel
# 2) Logging of all relevat options and results to file (either csv or text from std output)
# 3) Main training logbook to store all data to be used for model selection and hyperparameter tuning, this should be "per project"
# 4) Training mode: k-fold cross validation leveraging scikit-learn

@dataclass(frozen=True)
class ModelTrainingManagerConfig():
    '''Configuration dataclass for ModelTrainingManager class. Contains all parameters ModelTrainingManager accepts as configuration.'''
    
    # DATA fields with default values
    initial_lr: float = 1e-4
    lr_scheduler: Any = None
    optim_momentum: float = 0.5  # Momentum value for SGD optimizer
    
    # Define default optimizer as Adam
    optimizer: Any = torch.optim.Adam(lr=initial_lr)

    # DEVNOTE: dataclass generates __init__() automatically
    # Same goes for __repr()__ for printing and __eq()__ for equality check methods

    def getConfigDict(self) -> dict:
        '''Method to return the dataclass as dictionary'''
        return asdict(self)

    #def display(self) -> None:
    #    print('ModelTrainingManager configuration parameters:\n\t', self.getConfig())

    @classmethod
    # DEVNOTE: classmethod is like static methods, but apply to the class itself and passes it implicitly as the first argument
    def load_from_yaml(cls, yamlFile: Union[str, IO]) -> 'ModelTrainingManagerConfig':
        '''Method to load configuration parameters from a yaml file containing configuration dictionary'''

        if isinstance(yamlFile, str):
            with open(yamlFile, 'r') as file:

                # TODO: VALIDATE SCHEMA

                # Parse yaml file to dictionary
                configDict = yaml.safe_load(file)
        else:

            # TODO: VALIDATE SCHEMA

            configDict = yaml.safe_load(yamlFile)

        # Call load_from_dict() method
        return cls.load_from_dict(configDict)

    @classmethod
    def load_from_dict(cls, configDict: dict) -> 'ModelTrainingManagerConfig':
        '''Method to load configuration parameters from a dictionary'''

        # Get all field names from the class
        fieldNames = {f.name for f in fields(cls)}
        # Get fields in configuration dictionary
        missingFields = fieldNames - configDict.keys()

        # Check if any required field is missing (those without default values)
        requiredFields = {f.name for f in fields(
            cls) if f.default is MISSING and f.default_factory is MISSING}
        missingRequired = requiredFields & missingFields

        if missingRequired:
            raise ValueError(f"Config dict is missing required fields: {missingRequired}")
        
        # Build initialization arguments for class (using autogen __init__() method)
        # All fields not specified by configDict are initialized as default from cls values
        initArgs = {key: configDict.get( key, getattr(cls, key)) for key in fieldNames}
        
        # Return instance of class with attributes defined from dictionary
        return cls(**initArgs)
    
    @staticmethod
    def getConfigParamsNames(cls) -> list:
        '''Method to return the names of all parameters in the configuration class'''
        return [f.name for f in fields(cls)]

# %% ModelTrainingManager class - 24-07-2024
class ModelTrainingManager(ModelTrainingManagerConfig):
    '''Class to manage training and validation of PyTorch models using specified datasets and loss functions.'''
    
    def __init__(self, model: nn.Module, lossFcn: nn.Module, 
                 optimizer:Union[optim.Optimizer, int], options: Union[ModelTrainingManagerConfig, dict, str]) -> None:
        '''Constructor for TrainAndValidationManager class. Initializes model, loss function, optimizer and training/validation options.'''
        
        if isinstance(options, str):
            # Initialize ModelTrainingManagerConfig base instance from yaml file
            super().load_from_yaml(options)

        elif isinstance(options, dict):
            # Initialize ModelTrainingManagerConfig base instance from dictionary
            super().load_from_dict(options)

        elif isinstance(options, ModelTrainingManagerConfig):
            # Initialize ModelTrainingManagerConfig base instance from ModelTrainingManagerConfig instance
            raise NotImplementedError('Construction from ModelTrainingManagerConfig instance not supported yet.')


        # Define ModelTrainingManager attributes
        self.model = model
        self.lossFcn = lossFcn

        # Optimizer --> # TODO: check how to modify learning rate and momentum while training
        if isinstance(optimizer, optim.Optimizer):
            self.optimizer = optimizer

        elif isinstance(optimizer, int):
            if optimizer == 0:
                optimizer = torch.optim.SGD(
                     self.model.parameters(), lr=self.initial_lr, momentum=self.momentumValue)
            elif optimizer == 1:
                optimizer = torch.optim.Adam(
                     self.model.parameters(), lr=self.learnRate)
            else:
                raise ValueError(
                     'Optimizer type not recognized. Use either 0 for SGD or 1 for Adam.')
        else:
            raise ValueError(
                'Optimizer must be either an instance of torch.optim.Optimizer or an integer representing the optimizer type.')

        # Define training and validation options

    def LoadDatasets(self, dataloaderIndex: dict):
        '''Method to load datasets from dataloaderIndex and use them depending on the specified criterion (e.g. "order", "merge)'''
        # TODO: Load all datasets from dataloaderIndex and use them depending on the specified criterion (e.g. "order", "merge)
        pass

    def GetTracedModel(self):
        pass


# LEGACY FUNCTIONS - 18/09/2024
# %% Function to perform one step of training of a model using dataset and specified loss function - 04-05-2024
# Updated by PC 04-06-2024

def TrainModel(dataloader: DataLoader, model: nn.Module, lossFcn: nn.Module,
               optimizer, epochID: int, device=GetDevice(), taskType: str = 'classification', lr_scheduler=None,
               swa_scheduler=None, swa_model=None, swa_start_epoch: int = 15) -> Union[float, int]:
    '''Function to perform one step of training of a model using dataset and specified loss function'''
    model.train()  # Set model instance in training mode ("informing" backend that the training is going to start)

    counterForPrint = np.round(len(dataloader)/75)
    numOfUpdates = 0

    if swa_scheduler is not None or lr_scheduler is not None:
        mlflow.log_metric(
            'Learning rate', optimizer.param_groups[0]['lr'], step=epochID)

    print('Starting training loop using learning rate: {:.11f}'.format(
        optimizer.param_groups[0]['lr']))

    # Recall that enumerate gives directly both ID and value in iterable object
    for batchCounter, (X, Y) in enumerate(dataloader):

        # Get input and labels and move to target device memory
        # Define input, label pairs for target device
        X, Y = X.to(device), Y.to(device)

        # Perform FORWARD PASS to get predictions
        predVal = model(X)  # Evaluate model at input
        # Evaluate loss function to get loss value (this returns loss function instance, not a value)
        trainLossOut = lossFcn(predVal, Y)

        if isinstance(trainLossOut, dict):
            trainLoss = trainLossOut.get('lossValue')
            keys = [key for key in trainLossOut.keys() if key != 'lossValue']
            # Log metrics to MLFlow after converting dictionary entries to float
            mlflow.log_metrics({key: value.item() if isinstance(
                value, torch.Tensor) else value for key, value in trainLossOut.items()}, step=numOfUpdates)

        else:
            trainLoss = trainLossOut
            keys = []

        # Perform BACKWARD PASS to update parameters
        trainLoss.backward()  # Compute gradients
        optimizer.step()      # Apply gradients from the loss
        optimizer.zero_grad()  # Reset gradients for next iteration

        numOfUpdates += 1

        if batchCounter % counterForPrint == 0:  # Print loss value
            trainLoss, currentStep = trainLoss.item(), (batchCounter + 1) * len(X)
            # print(f"Training loss value: {trainLoss:>7f}  [{currentStep:>5d}/{size:>5d}]")
            # if keys != []:
            #    print("\t",", ".join([f"{key}: {trainLossOut[key]:.4f}" for key in keys]))    # Update learning rate if scheduler is provided
    # Perform step of SWA if enabled
    if swa_model is not None and epochID >= swa_start_epoch:
        # Update SWA model parameters
        swa_model.train()
        swa_model.update_parameters(model)
        # Update SWA scheduler
        # swa_scheduler.step()
        # Update batch normalization layers for swa model
        torch.optim.swa_utils.update_bn(dataloader, swa_model, device=device)
    # else:
    if lr_scheduler is not None:
        lr_scheduler.step()
        print('\n')
        print('Learning rate modified to: ', lr_scheduler.get_last_lr())
        print('\n')

    return trainLoss, numOfUpdates

# %% Function to validate model using dataset and specified loss function - 04-05-2024
# Updated by PC 04-06-2024
def ValidateModel(dataloader: DataLoader, model: nn.Module, lossFcn: nn.Module, device=GetDevice(), taskType: str = 'classification') -> Union[float, dict]:
    '''Function to validate model using dataset and specified loss function'''
    # Get size of dataset (How many samples are in the dataset)
    size = len(dataloader.dataset)

    model.eval()  # Set the model in evaluation mode
    validationLoss = 0  # Accumulation variables
    batchMaxLoss = 0

    validationData = {}  # Dictionary to store validation data

    # Initialize variables based on task type
    if taskType.lower() == 'classification':
        correctOuputs = 0

    elif taskType.lower() == 'regression':
        avgRelAccuracy = 0.0
        avgAbsAccuracy = 0.0

    elif taskType.lower() == 'custom':
        print('TODO')

    with torch.no_grad():  # Tell torch that gradients are not required
        # TODO: try to modify function to perform one single forward pass for the entire dataset

        # Backup the original batch size
        original_dataloader = dataloader
        original_batch_size = dataloader.batch_size

        # Temporarily initialize a new dataloader for validation
        allocMem = torch.cuda.memory_allocated(0)
        freeMem = torch.cuda.get_device_properties(
            0).total_memory - torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        estimated_memory_per_sample = allocMem / original_batch_size
        newBathSizeTmp = min(
            round(0.5 * freeMem / estimated_memory_per_sample), 2048)

        dataloader = DataLoader(
            dataloader.dataset, batch_size=newBathSizeTmp, shuffle=False, drop_last=False)
        lossTerms = {}
        numberOfBatches = len(dataloader)

        for X, Y in dataloader:
            # Get input and labels and move to target device memory
            X, Y = X.to(device), Y.to(device)

            # Perform FORWARD PASS
            predVal = model(X)  # Evaluate model at input
            # Evaluate loss function and accumulate
            tmpLossVal = lossFcn(predVal, Y)

            if isinstance(tmpLossVal, dict):
                tmpVal = tmpLossVal.get('lossValue').item()

                validationLoss += tmpVal
                if batchMaxLoss < tmpVal:
                    batchMaxLoss = tmpVal

                keys = [key for key in tmpLossVal.keys() if key != 'lossValue']
                # Sum all loss terms for each batch if present in dictionary output
                for key in keys:
                    lossTerms[key] = lossTerms.get(
                        key, 0) + tmpLossVal[key].item()
            else:

                validationLoss += tmpLossVal.item()
                if batchMaxLoss < tmpLossVal.item():
                    batchMaxLoss = tmpLossVal.item()

            validationData['WorstLossAcrossBatches'] = batchMaxLoss

            if taskType.lower() == 'classification':
                # Determine if prediction is correct and accumulate
                # Explanation: get largest output logit (the predicted class) and compare to Y.
                # Then convert to float and sum over the batch axis, which is not necessary if size is single prediction
                correctOuputs += (predVal.argmax(1) ==
                                  Y).type(torch.float).sum().item()

            # elif taskType.lower() == 'regression':
            #    #print('TODO')
            # elif taskType.lower() == 'custom':
            #    print('TODO')
            # predVal = model(dataX) # Evaluate model at input
            # validationLoss += lossFcn(predVal, dataY).item() # Evaluate loss function and accumulate

    # Log additional metrics to MLFlow if any
    if lossTerms != {}:
        lossTerms = {key: (value / numberOfBatches)
                     for key, value in lossTerms.items()}
        mlflow.log_metrics(lossTerms, step=0)

    # Restore the original batch size
    dataloader = original_dataloader

    # EXPERIMENTAL: Try to perform one single forward pass for the entire dataset (MEMORY BOUND)
    # with torch.no_grad():
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
        validationLoss /= numberOfBatches  # Compute batch size normalized loss value
        correctOuputs /= size  # Compute percentage of correct classifications over batch size
        print(
            f"\n VALIDATION ((Classification) Accuracy: {(100*correctOuputs):>0.2f}%, Avg loss: {validationLoss:>8f} \n")

    elif taskType.lower() == 'regression':
        # print('TODO')
        correctOuputs = None

        if isinstance(tmpLossVal, dict):
            keys = [key for key in tmpLossVal.keys() if key != 'lossValue']

        validationLoss /= numberOfBatches
        print(
            f"\n VALIDATION (Regression) Avg loss: {validationLoss:>0.5f}, Max batch loss: {batchMaxLoss:>0.5f}\n")
        # print(f"Validation (Regression): \n Avg absolute accuracy: {avgAbsAccuracy:>0.1f}, Avg relative accuracy: {(100*avgRelAccuracy):>0.1f}%, Avg loss: {validationLoss:>8f} \n")

    elif taskType.lower() == 'custom':
        print('TODO')

    return validationLoss, validationData
    # TODO: add command for Tensorboard here


# %% TRAINING and VALIDATION template function - 04-06-2024
def TrainAndValidateModel(dataloaderIndex: DataloaderIndex, model:nn.Module, lossFcn: nn.Module, optimizer, options:dict={}):
    # NOTE: is the default dictionary considered as "single" object or does python perform a merge of the fields?

    # TODO: For merging of options: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
    # if options is None:
    #    options = {}
    #
    # Merge user-provided options with default options
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
    # NOTE: Classification is not well developed (July, 2024)
    taskType = options.get('taskType', 'regression')
    device = options.get('device', GetDevice())
    numOfEpochs = options.get('epochs', 10)
    enableSave = options.get('saveCheckpoints', True)
    checkpointDir = options.get('checkpointsOutDir', './checkpoints')
    modelName = options.get('modelName', 'trainedModel')
    lossLogName = options.get('lossLogName', 'Loss_value')
    epochStart = options.get('epochStart', 0)

    swa_scheduler = options.get('swa_scheduler', None)
    swa_model = options.get('swa_model', None)
    swa_start_epoch = options.get('swa_start_epoch', 15)

    child_run = None
    child_run_name = None
    parent_run = mlflow.active_run()
    parent_run_name = parent_run.info.run_name

    lr_scheduler = options.get('lr_scheduler', None)
    # Default early stopping for regression: "minimize" direction
    # early_stopper = options.get('early_stopper', early_stopping=EarlyStopping(monitor="lossValue", patience=5, verbose=True, mode="min"))
    early_stopper = options.get('early_stopper', None)

    # Get Torch dataloaders
    trainingDataset = dataloaderIndex.getTrainloader()
    validationDataset = dataloaderIndex.getValidationLoader()

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

    # input('\n-------- PRESS ENTER TO START TRAINING LOOP --------\n')
    trainLossHistory = np.zeros(numOfEpochs)
    validationLossHistory = np.zeros(numOfEpochs)

    numOfUpdates = 0
    bestValidationLoss = 1E10
    bestSWAvalidationLoss = 1E10

    # Deep copy the initial state of the model and move it to the CPU
    bestModel = copy.deepcopy(model).to('cpu')
    bestEpoch = epochStart

    if swa_model != None:
        bestSWAmodel = copy.deepcopy(model).to('cpu')

    # TRAINING and VALIDATION LOOP
    for epochID in range(numOfEpochs):

        print(
            f"\n\t\t\tTRAINING EPOCH: {epochID + epochStart} of {epochStart + numOfEpochs-1}\n-------------------------------")
        # Do training over all batches
        trainLossHistory[epochID], numOfUpdatesForEpoch = TrainModel(trainingDataset, model, lossFcn, optimizer, epochID, device,
                                                                     taskType, lr_scheduler, swa_scheduler, swa_model, swa_start_epoch)
        numOfUpdates += numOfUpdatesForEpoch
        print('Current total number of updates: ', numOfUpdates)

        # Do validation over all batches
        validationLossHistory[epochID], validationData = ValidateModel(
            validationDataset, model, lossFcn, device, taskType)

        # If validation loss is better than previous best, update best model
        if validationLossHistory[epochID] < bestValidationLoss:
            # Replace best model with current model
            bestModel = copy.deepcopy(model).to('cpu')
            bestEpoch = epochID + epochStart
            bestValidationLoss = validationLossHistory[epochID]

            bestModelData = {'model': bestModel, 'epoch': bestEpoch,
                             'validationLoss': bestValidationLoss}

        print(
            f"Current best model found at epoch: {bestEpoch} with validation loss: {bestValidationLoss}")

        # SWA handling: if enabled, evaluate validation loss of SWA model, then decide if to update or reset
        if swa_model != None and epochID >= swa_start_epoch:

            # Verify swa_model on the validation dataset
            swa_model.eval()
            swa_validationLoss, _ = ValidateModel(
                validationDataset, swa_model, lossFcn, device, taskType)
            swa_model.train()
            print(
                f"Current SWA model found at epoch: {epochID} with validation loss: {swa_validationLoss}")

            # if swa_validationLoss < bestSWAvalidationLoss:
            # Update best SWA model
            bestSWAvalidationLoss = swa_validationLoss
            bestSWAmodel = copy.deepcopy(swa_model).to('cpu')
            swa_has_improved = True
            # else:
            #    # Reset to previous best model
            #    swa_model = copy.deepcopy(bestSWAmodel).to(device)
            #    swa_has_improved = False

            # Log data to mlflow by opening children run

            if child_run_name is None and child_run is None:
                child_run_name = parent_run_name + '-SWA'
                child_run = mlflow.start_run(
                    run_name=child_run_name, nested=True)
            mlflow.log_metric('SWA Best validation loss', bestSWAvalidationLoss,
                              step=epochID + epochStart, run_id=child_run.info.run_id)
        else:
            swa_has_improved = False

        # Re-open parent run scope
        mlflow.start_run(run_id=parent_run.info.run_id, nested=True)

        # Log parent run data
        mlflow.log_metric('Training loss - ' + lossLogName,
                          trainLossHistory[epochID], step=epochID + epochStart)
        mlflow.log_metric('Validation loss - ' + lossLogName,
                          validationLossHistory[epochID], step=epochID + epochStart)

        if 'WorstLossAcrossBatches' in validationData.keys():
            mlflow.log_metric('Validation Worst Loss across batches',
                              validationData['WorstLossAcrossBatches'], step=epochID + epochStart)

        if enableSave:
            if not (os.path.isdir(checkpointDir)):
                os.mkdir(checkpointDir)

            exampleInput = GetSamplesFromDataset(validationDataset, 1)[0][0].reshape(
                1, -1)  # Get single input sample for model saving
            modelSaveName = os.path.join(
                checkpointDir, modelName + '_' + AddZerosPadding(epochID + epochStart, stringLength=4))
            SaveTorchModel(model, modelSaveName, saveAsTraced=True,
                           exampleInput=exampleInput, targetDevice=device)

            if swa_model != None and swa_has_improved:
                swa_model.eval()
                SaveTorchModel(swa_model, modelSaveName + '_SWA', saveAsTraced=True,
                               exampleInput=exampleInput, targetDevice=device)
                swa_model.train()

        # %% MODEL PREDICTION EXAMPLES
        examplePrediction, exampleLosses, inputSampleTensor, labelsSampleTensor = EvaluateModel(
            validationDataset, model, lossFcn, device, 20)

        if swa_model is not None and epochID >= swa_start_epoch:
            # Test prediction of SWA model on the same input samples
            swa_model.eval()
            swa_examplePrediction, swa_exampleLosses, _, _ = EvaluateModel(
                validationDataset, swa_model, lossFcn, device, 20, inputSampleTensor, labelsSampleTensor)
            swa_model.train()

        # mlflow.log_artifacts('Prediction samples: ', validationLossHistory[epochID])

        # mlflow.log_param(f'ExamplePredictionList', list(examplePrediction))
        # mlflow.log_param(f'ExampleLosses', list(exampleLosses))

        print('\n  Random Sample predictions from validation dataset:\n')

        for id in range(examplePrediction.shape[0]):

            formatted_predictions = ['{:.5f}'.format(
                num) for num in examplePrediction[id, :]]
            formatted_loss = '{:.5f}'.format(exampleLosses[id])
            print(
                f'\tPrediction: {formatted_predictions} --> Loss: {formatted_loss}')

        print('\t\t Average prediction loss: {avgPred}\n'.format(
            avgPred=torch.mean(exampleLosses)))

        if swa_model != None and epochID >= swa_start_epoch:
            for id in range(examplePrediction.shape[0]):
                formatted_predictions = ['{:.5f}'.format(
                    num) for num in swa_examplePrediction[id, :]]
                formatted_loss = '{:.5f}'.format(swa_exampleLosses[id])
                print(
                    f'\tSWA Prediction: {formatted_predictions} --> Loss: {formatted_loss}')

            print('\t\t SWA Average prediction loss: {avgPred}\n'.format(
                avgPred=torch.mean(swa_exampleLosses)))

        # Perform step of Early stopping if enabled
        if early_stopper is not None:
            print('Early stopping NOT IMPLEMENTED. Skipping...')
            # early_stopper.step(validationLossHistory[epochID])
            # if early_stopper.early_stop:
            #    mlflow.end_run(status='KILLED')
            #    print('Early stopping triggered at epoch: {epochID}'.format(epochID=epochID))
            #    break
            # earlyStopping(validationLossHistory[epochID], model, bestModelData, options)
    if swa_model != None and epochID >= swa_start_epoch:
        # End nested child run
        mlflow.end_run(status='FINISHED')
    # End parent run
    mlflow.end_run(status='FINISHED')

    if swa_model is not None:
        bestModelData['swa_model'] = bestSWAmodel

    return bestModelData, trainLossHistory, validationLossHistory, inputSampleTensor

# %% Model evaluation function on a random number of samples from dataset - 06-06-2024
# Possible way to solve the issue of having different cost function terms for training and validation --> add setTrain and setEval methods to switch between the two


def EvaluateModel(dataloader: DataLoader, model: nn.Module, lossFcn: nn.Module, device=GetDevice(), numOfSamples: int = 10,
                  inputSample: torch.tensor = None, labelsSample: torch.tensor = None) -> np.array:
    '''Torch model evaluation function to perform inference using either specified input samples or input dataloader'''
    model.eval()  # Set model in prediction mode
    with torch.no_grad():
        if inputSample is None and labelsSample is None:
            # Get some random samples from dataloader as list
            extractedSamples = GetSamplesFromDataset(dataloader, numOfSamples)

            # Create input array as torch tensor
            X = torch.zeros(len(extractedSamples),
                            extractedSamples[0][0].shape[0])
            Y = torch.zeros(len(extractedSamples),
                            extractedSamples[0][1].shape[0])

            # inputSampleList = []
            for id, (inputVal, labelVal) in enumerate(extractedSamples):
                X[id, :] = inputVal
                Y[id, :] = labelVal

            # inputSampleList.append(inputVal.reshape(1, -1))

            # Perform FORWARD PASS
            examplePredictions = model(X.to(device))  # Evaluate model at input

            # Compute loss for each input separately
            exampleLosses = torch.zeros(examplePredictions.size(0))

            examplePredictionList = []
            for id in range(examplePredictions.size(0)):

                # Get prediction and label samples
                examplePredictionList.append(
                    examplePredictions[id, :].reshape(1, -1))
                labelSample = Y[id, :].reshape(1, -1)

                # Evaluate loss function
                outLossVar = lossFcn(examplePredictionList[id].to(
                    device), labelSample.to(device))

                if isinstance(outLossVar, dict):
                    exampleLosses[id] = outLossVar.get('lossValue').item()
                else:
                    exampleLosses[id] = outLossVar.item()

        elif inputSample is not None and labelsSample is not None:
            # Perform FORWARD PASS # NOTE: NOT TESTED
            X = inputSample
            Y = labelsSample

            examplePredictions = model(X.to(device))  # Evaluate model at input

            exampleLosses = torch.zeros(examplePredictions.size(0))
            examplePredictionList = []

            for id in range(examplePredictions.size(0)):

                # Get prediction and label samples
                examplePredictionList.append(
                    examplePredictions[id, :].reshape(1, -1))
                labelSample = Y[id, :].reshape(1, -1)

                # Evaluate loss function
                outLossVar = lossFcn(examplePredictionList[id].to(
                    device), labelSample.to(device))

                if isinstance(outLossVar, dict):
                    exampleLosses[id] = outLossVar.get('lossValue').item()
                else:
                    exampleLosses[id] = outLossVar.item()
        else:
            raise ValueError(
                'Either both inputSample and labelsSample must be provided or neither!')

        return examplePredictions, exampleLosses, X.to(device), Y.to(device)
