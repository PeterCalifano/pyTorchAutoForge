# TODO Add yaml interface for training, compatible with mlflow and optuna
# The idea is to let the user specify all the parameters in a yaml file, which is then loaded and used
# to set the configuration class. Default values are specified as the class defaults.
# Loading methods only modify the parameters the user has specified


from typing import Optional, Any, Union, IO
import torch
import mlflow
import os, sys, traceback
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict, fields, Field, MISSING

from pyTorchAutoForge.datasets import DataloaderIndex
from pyTorchAutoForge.utils.utils import GetDevice, AddZerosPadding, GetSamplesFromDataset
from pyTorchAutoForge.api.torch import SaveTorchModel
from pyTorchAutoForge.optimization import CustomLossFcn

# import datetime
import yaml
import copy
from enum import Enum

# Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim


class TaskType(Enum):
    '''Enum class to define task types for training and validation'''
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    CUSTOM = 'custom'


# %% Training and validation manager class - 22-06-2024 (WIP)
# TODO: Features to include:
# 1) Multi-process/multi-threading support for training and validation of multiple models in parallel
# 2) Logging of all relevat config and results to file (either csv or text from std output)
# 3) Main training logbook to store all data to be used for model selection and hyperparameter tuning, this should be "per project"
# 4) Training mode: k-fold cross validation leveraging scikit-learn

@dataclass(frozen=False)
class ModelTrainingManagerConfig():
    '''Configuration dataclass for ModelTrainingManager class. Contains all parameters ModelTrainingManager accepts as configuration.'''

    # REQUIRED fields
    tasktype: TaskType # Task type for training and validation --> How to enforce the definition of this?

    # FIELDS with DEFAULTS
    # Optimization strategy
    num_of_epochs: int = 10  # Number of epochs for training
    keep_best: bool = True  # Keep best model during training
    enable_early_stop: bool = False  # Enable early stopping
    checkpointDir = "./checkpoints"  # Directory to save model checkpoints
    modelName = "trained_model"      # Name of the model to be saved

    # Logging
    mlflow_logging: bool = True  # Enable MLFlow logging
    eval_example: bool = False  # Evaluate example input during training

    # Optimization parameters
    lr_scheduler: Any = None 
    initial_lr: float = 1e-4
    optim_momentum: float = 0.5  # Momentum value for SGD optimizer
    optimizer: Any = torch.optim.Adam # optimizer class
    
    # Hardware settings
    device: str = GetDevice()  # Default device is GPU if available


    def __copy__(self, instanceToCopy: 'ModelTrainingManagerConfig') -> 'ModelTrainingManagerConfig':
        """
        Create a shallow copy of the ModelTrainingManagerConfig instance.

        Returns:
            ModelTrainingManagerConfig: A new instance of ModelTrainingManagerConfig with the same configuration.
        """
        return self.__init__(**instanceToCopy.getConfigDict())
    
    # DEVNOTE: dataclass generates __init__() automatically
    # Same goes for __repr()__ for printing and __eq()__ for equality check methods

    def getConfigDict(self) -> dict:
        """
        Returns the configuration of the model training manager as a dictionary.

        This method converts the instance attributes of the model training manager
        into a dictionary format using the `asdict` function.

        Returns:
            dict: A dictionary containing the attributes of the model training manager.
        """
        return asdict(self)

    # def display(self) -> None:
    #    print('ModelTrainingManager configuration parameters:\n\t', self.getConfig())

    @classmethod
    # DEVNOTE: classmethod is like static methods, but apply to the class itself and passes it implicitly as the first argument
    def load_from_yaml(cls, yamlFile: Union[str, IO]) -> 'ModelTrainingManagerConfig':
        '''Method to load configuration parameters from a yaml file containing configuration dictionary'''

        if isinstance(yamlFile, str):
            # Check if file exists
            if not os.path.isfile(yamlFile):
                raise FileNotFoundError(f"File not found: {yamlFile}")
            
            with open(yamlFile, 'r') as file:

                # TODO: VALIDATE SCHEMA

                # Parse yaml file to dictionary
                configDict = yaml.safe_load(file)
        else:

            # TODO: VALIDATE SCHEMA
            configDict = yaml.safe_load(yamlFile)

        # Call load_from_dict() method
        return cls.load_from_dict(configDict)

    @classmethod # Why did I defined this class instead of using the __init__ method for dataclasses?
    def load_from_dict(cls, configDict: dict) -> 'ModelTrainingManagerConfig':
        """
        Load configuration parameters from a dictionary and return an instance of the class. If attribute is not present, default/already assigned value is used unless required.

        Args:
            configDict (dict): A dictionary containing configuration parameters.

        Returns:
            ModelTrainingManagerConfig: An instance of the class with attributes defined from the dictionary.

        Raises:
            ValueError: If the configuration dictionary is missing any required fields.
        """

        # Get all field names from the class
        fieldNames = {f.name for f in fields(cls)}
        # Get fields in configuration dictionary
        missingFields = fieldNames - configDict.keys()

        # Check if any required field is missing (those without default values)
        requiredFields = {f.name for f in fields(
            cls) if f.default is MISSING and f.default_factory is MISSING}
        missingRequired = requiredFields & missingFields

        if missingRequired:
            raise ValueError(
                f"Config dict is missing required fields: {missingRequired}")

        # Build initialization arguments for class (using autogen __init__() method)
        # All fields not specified by configDict are initialized as default from cls values
        initArgs = {key: configDict.get(key, getattr(cls, key))
                    for key in fieldNames}

        # Return instance of class with attributes defined from dictionary
        return cls(**initArgs)

    @classmethod
    def getConfigParamsNames(self) -> list:
        '''Method to return the names of all parameters in the configuration class'''
        return [f.name for f in fields(self)]
    


# %% ModelTrainingManager class - 24-07-2024
class ModelTrainingManager(ModelTrainingManagerConfig):
    def __init__(self, model: Union[nn.Module], lossFcn: Union[nn.Module, CustomLossFcn], config: Union[ModelTrainingManagerConfig, dict, str], optimizer: Union[optim.Optimizer, int, None] = None, dataLoaderIndex: Optional[DataloaderIndex] = None) -> None:
        """
        Initializes the ModelTrainingManager class.

        Parameters:
        model (nn.Module): The neural network model to be trained.
        lossFcn (Union[nn.Module, CustomLossFcn]): The loss function to be used during training.
        config (Union[ModelTrainingManagerConfig, dict, str]): Configuration config for training. Can be a ModelTrainingManagerConfig instance, a dictionary, or a path to a YAML file.
        optimizer (Union[optim.Optimizer, int, None], optional): The optimizer to be used for training. Can be an instance of torch.optim.Optimizer, an integer (0 for SGD, 1 for Adam), or None. Defaults to None.

        Raises:
        ValueError: If the optimizer is not an instance of torch.optim.Optimizer or an integer representing the optimizer type, or if the optimizer ID is not recognized.
        """
        # Load configuration parameters from config
        if isinstance(config, str):
            # Initialize ModelTrainingManagerConfig base instance from yaml file
            super().load_from_yaml(config)

        elif isinstance(config, dict):
            # Initialize ModelTrainingManagerConfig base instance from dictionary
            super().load_from_dict(config) # This method only copies the attributes present in the dictionary, which may be a subset.

        elif isinstance(config, ModelTrainingManagerConfig):
            # Initialize ModelTrainingManagerConfig base instance from ModelTrainingManagerConfig instance
            super().__init__(**config.getConfigDict())  # Call init of parent class for shallow copy
            
        # Initialize ModelTrainingManager-specific attributes
        self.model = (model).to(self.device)
        self.bestModel = None
        self.lossFcn = lossFcn
        self.trainingDataloader = None
        self.validationDataloader = None
        self.trainingDataloaderSize = 0
        self.currentEpoch = 0
        self.numOfUpdates = 0

        self.currentTrainingLoss = None
        self.currentValidationLoss = None
        self.currentMlflowRun = mlflow.active_run() # Returns None if no active run

        self.current_lr = self.initial_lr

        # Initialize dataloaders if provided
        if dataLoaderIndex is not None:
            self.setDataloaders(dataLoaderIndex)

        # Handle override of optimizer inherited from ModelTrainingManagerConfig
        if optimizer is not None:
            if isinstance(optimizer, optim.Optimizer):
                self.reinstantiate_optimizer_()
            elif isinstance(optimizer, int) or issubclass(optimizer, optim.Optimizer):
                self.define_optimizer_(optimizer)
            else:
                raise ValueError('Optimizer must be either an instance of torch.optim.Optimizer or an integer representing the optimizer type.')
        else:
            if isinstance(self.optimizer, optim.Optimizer):
                # Redefine optimizer class as workaround for weird python behavior (no update applied to model)
                self.reinstantiate_optimizer_()
            else: 
                raise ValueError('Optimizer must be specified either in the ModelTrainingManagerConfig as torch.optim.Optimizer instance or as an argument in __init__ of this class!')

    def define_optimizer_(self, optimizer: Union[optim.Optimizer, int]) -> None:
        """
        Define and set the optimizer for the model training.

        Parameters:
        optimizer (Union[torch.optim.Optimizer, int]): The optimizer to be used for training. 
            It can be an instance of a PyTorch optimizer or an integer identifier.
            - If 0 or torch.optim.SGD, the Stochastic Gradient Descent (SGD) optimizer will be used.
            - If 1 or torch.optim.Adam, the Adam optimizer will be used.

        Raises:
        ValueError: If the optimizer ID is not recognized (i.e., not 0 or 1).
        """
        if optimizer == 0 or optimizer == torch.optim.SGD:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.initial_lr, momentum=self.momentumValue)
        elif optimizer == 1 or optimizer == torch.optim.Adam:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.initial_lr)
            
    def reinstantiate_optimizer_(self):
        """
        Reinstantiates the optimizer with the same hyperparameters but with the current model parameters.
        """
        optim_class = self.optimizer.__class__
        optim_params = self.optimizer.param_groups[0]
        optimizer_hyperparams = {key: value for key, value in optim_params.items() if key != 'params'}
        self.optimizer = optim_class(self.model.parameters(), **optimizer_hyperparams)

        if self.lr_scheduler is not None:
            self.lr_scheduler.optimizer = self.optimizer
        
    def setDataloaders(self, dataloaderIndex: DataloaderIndex) -> None:
        """
        Sets the training and validation dataloaders using the provided DataloaderIndex.

        Args:
            dataloaderIndex (DataloaderIndex): An instance of DataloaderIndex that provides
                                               the training and validation dataloaders.
        """
        self.trainingDataloader = dataloaderIndex.getTrainLoader()
        self.validationDataloader = dataloaderIndex.getValidationLoader()
        self.trainingDataloaderSize = len(self.trainingDataloader)

    def getTracedModel(self):
        raise NotImplementedError('Method not implemented yet.')

    def trainModelOneEpoch_(self):
        '''Method to train the model using the specified datasets and loss function. Not intended to be called as standalone.'''
        
        if self.trainingDataloader is None:
            raise ValueError('No training dataloader provided.')
        
        # Set model instance in training mode (mainly for dropout and batch normalization layers)
        self.model.train()  # Set model instance in training mode ("informing" backend that the training is going to start)
        
        running_loss = 0.0
        #prev_model = copy.deepcopy(self.model)
        for batch_idx, (X, Y) in enumerate(self.trainingDataloader):

            # Get input and labels and move to target device memory
            # Define input, label pairs for target device
            X, Y = X.to(self.device), Y.to(self.device) # DEVNOTE: TBD if this goes here or if to move dataloader to device

            # Perform FORWARD PASS to get predictions
            predVal = self.model(X)  # Evaluate model at input, calls forward() method
            
            # Evaluate loss function to get loss value dictionary
            trainLossDict = self.lossFcn(predVal, Y)

            # Get loss value from dictionary
            trainLossVal = trainLossDict.get('lossValue') if isinstance(trainLossDict, dict) else trainLossDict

            # TODO: here one may log intermediate metrics at each update
            # if self.mlflow_logging:
            #     mlflow.log_metrics()

            # Update running value of loss for status bar at current epoch
            running_loss += trainLossVal.item()

            # Perform BACKWARD PASS to update parameters
            self.optimizer.zero_grad()  # Reset gradients for next iteration
            trainLossVal.backward()     # Compute gradients
            self.optimizer.step()       # Apply gradients from the loss
            self.numOfUpdates += 1

            # Calculate progress
            current_batch = batch_idx + 1
            progress = f"\tBatch {batch_idx+1}/{self.trainingDataloaderSize}, average loss: {running_loss / current_batch:.4f}, number of updates: {self.numOfUpdates}, current lr: {self.current_lr:.11f}"

            # Print progress on the same line
            sys.stdout.write('\r' + progress)
            sys.stdout.flush()

            # TODO: implement management of SWA model
            #if swa_model is not None and epochID >= swa_start_epoch:
        

        # DEBUG: 
        #print(f"\n\nDEBUG: Model parameters before and after optimizer step:")
        #for param1, param2 in zip(prev_model.parameters(), self.model.parameters()):
        #    if torch.equal(param1, param2) or param1 is param2:
        #        raise ValueError("Model parameters are the same after 1 epoch.")

        # Post epoch operations
        self.currentEpoch += 1

        return running_loss / current_batch

    def validateModel_(self):
        """Method to validate the model using the specified datasets and loss function. Not intended to be called as standalone."""
        if self.validationDataloader is None:
            raise ValueError('No validation dataloader provided.')
        
        self.model.eval()
        validationLossVal = 0.0  # Accumulation variables
        # batchMaxLoss = 0
        # validationData = {}  # Dictionary to store validation data

        # Backup the original batch size (TODO: TBC if it is useful)
        original_dataloader = self.validationDataloader
                
        # Temporarily initialize a new dataloader for validation
        newBathSizeTmp = 2 * self.validationDataloader.batch_size # TBC how to set this value

        tmpdataloader = DataLoader(
                                original_dataloader.dataset, 
                                batch_size=newBathSizeTmp, 
                                shuffle=False, 
                                drop_last=False, 
                                pin_memory=True,
                                num_workers=0
                                )
        
        numberOfBatches = len(tmpdataloader)
        dataset_size = len(tmpdataloader.dataset)

        with torch.no_grad():
            if self.tasktype == TaskType.CLASSIFICATION:

                if not(isinstance(self.lossFcn, torch.nn.CrossEntropyLoss)):
                    raise NotImplementedError('Current classification validation function only supports nn.CrossEntropyLoss.')
                
                correctPredictions = 0

                for X, Y in tmpdataloader:
                    # Get input and labels and move to target device memory
                    X, Y = X.to(self.device), Y.to(self.device)

                    # Perform FORWARD PASS
                    predVal = self.model(X)  # Evaluate model at input

                    # Evaluate loss function to get loss value dictionary
                    validationLossDict = self.lossFcn(predVal, Y)
                    validationLossVal += validationLossDict.get('lossValue') if isinstance(validationLossDict, dict) else validationLossDict.item()

                    # Evaluate how many correct predictions (assuming CrossEntropyLoss)
                    correctPredictions += (predVal.argmax(1) == Y).type(torch.float).sum().item()

                validationLossVal /= numberOfBatches  # Compute batch size normalized loss value
                correctPredictions /= dataset_size    # Compute percentage of correct classifications over dataset size
                print(f"\n\tValidation: classification accuracy: {(100*correctPredictions):>0.2f}%, average loss: {validationLossVal:>4f}\n")

                return validationLossVal, correctPredictions

            elif self.tasktype == TaskType.REGRESSION:
                
                for X, Y in tmpdataloader:
                    # Get input and labels and move to target device memory
                    X, Y = X.to(self.device), Y.to(self.device)

                    # Perform FORWARD PASS
                    predVal = self.model(X)  # Evaluate model at input

                    # Evaluate loss function to get loss value dictionary
                    validationLossDict = self.lossFcn(predVal, Y)

                    # Get loss value from dictionary
                    validationLossVal += validationLossDict.get('lossValue') if isinstance(validationLossDict, dict) else validationLossDict.item()

                validationLossVal /= numberOfBatches  # Compute batch size normalized loss value
                print(f"\n\tValidation: regression average loss: {validationLossVal:>4f}\n")

                return validationLossVal
            
            else:
                raise NotImplementedError('Custom task type not implemented yet.')
            
    def trainAndValidate(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        self.startMlflowRun()
        try:
            for epoch_num in range(self.num_of_epochs):

                print(f"\nTraining epoch: {epoch_num+1}/{self.num_of_epochs}:")
                # Update current learning rate
                self.current_lr = self.optimizer.param_groups[0]['lr']

                if self.mlflow_logging:
                    mlflow.log_metric('lr', self.current_lr, step=self.currentEpoch)

                # Perform training for one epoch
                tmpTrainLoss = self.trainModelOneEpoch_()                

                # Perform validation at current epoch
                tmpValidLoss = self.validateModel_()

                if isinstance(tmpValidLoss, tuple):
                    tmpValidLoss = tmpValidLoss[0]

                # Execute post-epoch operations
                self.updateLerningRate()  # Update learning rate if scheduler is provided
                examplePrediction, exampleLoss = self.evalExample()        # Evaluate example if enabled

                if self.currentValidationLoss is None: # At epoch 0, set initial validation loss
                    self.currentValidationLoss = tmpValidLoss
                
                # Update stats if new best model found (independently of keep_best flag)
                if tmpValidLoss <= self.currentValidationLoss:
                    self.bestEpoch = epoch_num
                    self.bestValidationLoss = tmpValidLoss
                    noNewBestCounter = 0
                else:
                    noNewBestCounter += 1

                # "Keep best" strategy implementation (trainer will output the best overall model at cycle end)
                # DEVNOTE: this could go into a separate method
                if self.keep_best:
                    if tmpValidLoss <= self.currentValidationLoss:
                        self.bestModel = copy.deepcopy(self.model).to('cpu') # Transfer best model to CPU to avoid additional memory allocation on GPU
                
                # Update current training and validation loss values
                self.currentTrainingLoss = tmpTrainLoss
                self.currentValidationLoss = tmpValidLoss

                if self.mlflow_logging:
                    mlflow.log_metric('train_loss', self.currentTrainingLoss, step=self.currentEpoch)
                    mlflow.log_metric('validation_loss', self.currentValidationLoss, step=self.currentEpoch)

                # "Early stopping" strategy implementation
                if self.checkForEarlyStop(noNewBestCounter): 
                    break
            
            # Model saving code
            modelToSave = (self.bestModel if self.bestModel is not None else self.model).to('cpu')
            if self.keep_best:
                print('Best model saved from epoch: {best_epoch} with validation loss: {best_loss}'.format(best_epoch=self.bestEpoch, best_loss=self.bestValidationLoss))
            
            if not (os.path.isdir(self.checkpointDir)):
                os.mkdir(self.checkpointDir)

            exampleInput = GetSamplesFromDataset(self.validationDataloader, 1)[0][0].reshape(1, -1)  # Get single input sample for model saving

            modelSaveName = os.path.join(self.checkpointDir, self.modelName + '.pth')
            SaveTorchModel(modelToSave, modelSaveName, saveAsTraced=True, exampleInput=exampleInput, targetDevice='cpu')

        except Exception as e:

            if e is KeyboardInterrupt:
                print('ModelTrainingManager stopped execution due to KeyboardInterrupt. Run marked as KILLED.')
                if self.mlflow_logging:
                    mlflow.end_run(status='KILLED')
            else:
                print(f"Error during training and validation cycle: {e}")
                traceback.print_exc()
                if self.mlflow_logging:
                    mlflow.end_run(status='FAILED')

        
        # Post-training operations
        print('Training and validation cycle completed.')
        if self.mlflow_logging:
            mlflow.end_run(status='FINISHED')


    def evalExample(self) -> Union[torch.Tensor, None]:
        if self.eval_example:
            #exampleInput = GetSamplesFromDataset(self.validationDataloader, 1)[0][0].reshape(1, -1)
            #if self.mlflow_logging: # TBC, not sure it is useful
            #    # Log example input to mlflow
            #    mlflow.log_???('example_input', exampleInput)
            with torch.no_grad():
                examplePair = next(iter(self.validationDataloader))[0]

                X = examplePair[0].to(self.device)
                Y = examplePair[1].to(self.device)

                # Perform FORWARD PASS
                examplePredictions = self.model(X)  # Evaluate model at input

                # Compute loss for each input separately                
                outLossVar = self.lossFcn(examplePredictions, Y)
                
            # TODO (TBC): log example in mlflow?
            if self.mlflow_logging:
                print('TBC')

            formatted_predictions = [f"{sample:4.04f}" for sample in examplePredictions.reshape(-1).tolist()]
            print("Sample prediction: ", formatted_predictions, " with loss: ", outLossVar.item())

            return examplePredictions, outLossVar
        else:
            return None, None

    def updateLerningRate(self):
        if self.lr_scheduler is not None:
            # Perform step of learning rate scheduler if provided
            self.lr_scheduler.step()
            print('\nLearning rate changed: ${prev_lr} --> ${current_lr}\n'.format(prev_lr=self.current_lr, current_lr=self.lr_scheduler.get_last_lr()) )

            # Update current learning rate
            self.current_lr = self.lr_scheduler.get_last_lr()

    def checkForEarlyStop(self, counter: int) -> bool:
        """
        Checks if the early stopping criteria have been met.
        Parameters:
        counter (int): The current count of epochs or iterations without improvement.
        Returns:
        bool: True if early stopping criteria are met and training should stop, False otherwise.
        """
        returnValue = False

        if self.enable_early_stop:
            if counter >= self.early_stop_patience:
                print('Early stopping criteria met: ModelTrainingManager execution stop. Run marked as KILLED.')
                returnValue = True
                if self.mlflow_logging:
                    mlflow.end_run(status='KILLED')

        return returnValue
    
    def startMlflowRun(self):
        """
        Starts a new MLflow run if MLflow logging is enabled.

        If there is an active MLflow run, it ends the current run before starting a new one.
        Updates the current MLflow run to the newly started run.

        Prints messages indicating the status of the MLflow runs.

        Raises:
            Warning: If MLflow logging is disabled.
        """
        if self.mlflow_logging:
            if self.currentMlflowRun is not None:
                mlflow.end_run()
                print(('\nActive mlflow run {active_run} ended before creating new one.').format(
                    active_run=self.currentMlflowRun.info.run_name))

            mlflow.start_run()
            # Update current mlflow run
            self.currentMlflowRun = mlflow.active_run()
            print(('\nStarted new mlflow run witn name: {active_run}.').format(
                active_run=self.currentMlflowRun.info.run_name))
            
            # Log configuration parameters
            ModelTrainerConfigParamsNames = ModelTrainingManagerConfig.getConfigParamsNames()      
            #print("DEBUG:", ModelTrainerConfigParamsNames)
            mlflow.log_params({key: getattr(self, key) for key in ModelTrainerConfigParamsNames})
        
        #else:
        #    Warning('MLFlow logging is disabled. No run started.')

        



            


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
            dataloader.dataset, 
            batch_size=newBathSizeTmp, 
            shuffle=False, 
            drop_last=False, 
            pin_memory=True,
            num_workers=0)
        
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
                correctOuputs += (predVal.argmax(1) == Y).type(torch.float).sum().item()

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
            f"\n VALIDATION (Classification) Accuracy: {(100*correctOuputs):>0.2f}%, Avg loss: {validationLoss:>8f} \n")

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
def TrainAndValidateModel(dataloaderIndex: DataloaderIndex, model: nn.Module, lossFcn: nn.Module, optimizer, config: dict = {}):
    '''Function to train and validate a model using specified dataloaders and loss function'''
    # NOTE: is the default dictionary considered as "single" object or does python perform a merge of the fields?

    # TODO: For merging of config: https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
    # if config is None:
    #    config = {}
    #
    # Merge user-provided config with default config
    # combined_options = {**default_options, **config}
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

    # Setup config from input dictionary
    # NOTE: Classification is not developed (July, 2024)
    taskType = config.get('taskType', 'regression')
    device = config.get('device', GetDevice())
    numOfEpochs = config.get('epochs', 10)
    enableSave = config.get('saveCheckpoints', True)
    checkpointDir = config.get('checkpointsOutDir', './checkpoints')
    modelName = config.get('modelName', 'trainedModel')
    lossLogName = config.get('lossLogName', 'Loss_value')
    epochStart = config.get('epochStart', 0)

    swa_scheduler = config.get('swa_scheduler', None)
    swa_model = config.get('swa_model', None)
    swa_start_epoch = config.get('swa_start_epoch', 15)

    child_run = None
    child_run_name = None
    parent_run = mlflow.active_run()
    parent_run_name = parent_run.info.run_name

    lr_scheduler = config.get('lr_scheduler', None)
    # Default early stopping for regression: "minimize" direction
    # early_stopper = config.get('early_stopper', early_stopping=EarlyStopping(monitor="lossValue", patience=5, verbose=True, mode="min"))
    early_stopper = config.get('early_stopper', None)

    # Get Torch dataloaders
    trainingDataset = dataloaderIndex.getTrainLoader()
    validationDataset = dataloaderIndex.getValidationLoader()

    # Configure Tensorboard
    # if 'logDirectory' in config.keys():
    #    logDirectory = config['logDirectory']
    # else:
    #    currentTime = datetime.datetime.now()
    #    formattedTimestamp = currentTime.strftime('%d-%m-%Y_%H-%M') # Format time stamp as day, month, year, hour and minute
    #    logDirectory = './tensorboardLog_' + modelName + formattedTimestamp

    # if not(os.path.isdir(logDirectory)):
    #    os.mkdir(logDirectory)
    # tensorBoardWriter = ConfigTensorboardSession(logDirectory, portNum=tensorBoardPortNum)

    # If training is being restarted, attempt to load model
    if config['loadCheckpoint'] == True:
        raise NotImplementedError(
            'Training restart from checkpoint REMOVED. Not updated with mlflow yet.')
        model = LoadModelAtCheckpoint(
            model, config['checkpointsInDir'], modelName, epochStart)

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

        print(f"\n\t\t\tTRAINING EPOCH: {epochID + epochStart} of {epochStart + numOfEpochs-1}\n-------------------------------")
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
            # earlyStopping(validationLossHistory[epochID], model, bestModelData, config)
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
