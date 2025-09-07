"""
    _summary_

_extended_summary_

:raises FileNotFoundError: _description_
:raises ValueError: _description_
:raises ValueError: _description_
:raises ValueError: _description_
:raises ValueError: _description_
:raises ValueError: _description_
:raises TypeError: _description_
:raises NotImplementedError: _description_
:raises TypeError: _description_
:raises ValueError: _description_
:raises ValueError: _description_
:raises ValueError: _description_
:raises NotImplementedError: _description_
:raises NotImplementedError: _description_
:raises ValueError: _description_
:raises ValueError: _description_
:raises NotImplementedError: _description_
:raises optuna.TrialPruned: _description_
:raises optuna.TrialPruned: _description_
:raises optuna.TrialPruned: _description_
:raises optuna.TrialPruned: _description_
:raises ValueError: _description_
:raises NotImplementedError: _description_
:raises TypeError: _description_
:raises ValueError: _description_
:raises NotImplementedError: _description_
:raises NotImplementedError: _description_
:raises ValueError: _description_
:return: _description_
:rtype: _type_
"""

# TODO Add yaml interface for training, compatible with mlflow and optuna
# The idea is to let the user specify all the parameters in a yaml file, which is then loaded and used
# to set the configuration class. Default values are specified as the class defaults.
# Loading methods only modify the parameters the user has specified

# from warnings import deprecated
from math import isnan
from typing import Literal
import pprint
from typing import Any, IO
import torch
import mlflow
import optuna
import os
import sys
import time
import colorama
import glob
import traceback
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict, fields, field, MISSING
import math

from pyTorchAutoForge.datasets import DataloaderIndex, ImageAugmentationsHelper
from pyTorchAutoForge.utils import GetDeviceMulti, AddZerosPadding, GetSamplesFromDataset, ComputeModelParamsStorageSize
from pyTorchAutoForge.api.torch import SaveModel, LoadModel, AutoForgeModuleSaveMode
from pyTorchAutoForge.api.mlflow import RecursiveLogParamsInDict
from pyTorchAutoForge.optimization import CustomLossFcn

from inputimeout import inputimeout, TimeoutOccurred

# import datetime
import yaml
import copy
from enum import Enum

# Key class to use tensorboard with PyTorch. VSCode will automatically ask if you want to load tensorboard in the current session.
import torch.optim as optim
import inspect


class TaskType(Enum):
    '''Enum class to define task types for training and validation'''
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    SEGMENTATION = 'segmentation'
    CUSTOM = 'custom'


# %% Training and validation manager class - 22-06-2024 (WIP)
# TODO: Features to include:
# 1) Multi-process/multi-threading support for training and validation of multiple models in parallel
# 2) Logging of all relevat config and results to file (either csv or text from std output)
# 3) Main training logbook to store all data to be used for model selection and hyperparameter tuning, this should be "per project"
# 4) Training mode: k-fold cross validation leveraging scikit-learn


@dataclass
class LearnRateReductOnPlateauConfig():
    '''Configuration dataclass for learning rate reduction on plateau. Contains all parameters for learning rate reduction on plateau.'''
    lr_reduction_plateau_patience: int = 25  # Number of epochs to wait before reducing learning rate
    lr_reduct_factor: float = 0.2
    score_mode: Literal['min', 'max'] = 'min'
    score_eval_thr: float = 1e-3
    score_eval_thr_mode: Literal['rel', 'abs'] = 'rel'
    min_lr: float = 1E-8
    num_cooldown_epochs: int = 5

    def to_scheduler_kwargs(self) -> dict:
        return {
            "factor":           self.lr_reduct_factor,
            "patience":         self.lr_reduction_plateau_patience,
            "cooldown":         self.num_cooldown_epochs,
            "min_lr":           self.min_lr,
            "mode":             self.score_mode,
            "threshold":        self.score_eval_thr,
            "threshold_mode":   self.score_eval_thr_mode,
        }


@dataclass()
class ModelTrainingManagerConfig():  # TODO update to use BaseConfigClass
    '''
    Configuration dataclass for ModelTrainingManager class. Contains all parameters ModelTrainingManager accepts as configuration.
    '''

    # REQUIRED fields
    # Task type for training and validation
    tasktype: TaskType
    batch_size: int

    # DIFFERENTIABLE DATA AUGMENTATION
    data_augmentation_module: torch.nn.Sequential | ImageAugmentationsHelper | None = None
    augment_validation_data: bool = False
    enable_augs_autograd: bool = False

    # FIELDS with DEFAULTS
    # Optimization strategy
    num_of_epochs: int = 100  # Number of epochs for training
    current_epoch: int = 0
    keep_best: bool = True  # Keep best model during training
    enable_early_pruning: bool = False  # Enable early pruning
    pruning_patience: int = 50  # Number of epochs to wait before pruning
    # Number of batches to accumulate gradients before updating weights
    batch_accumulation_factor: int = 1
    # Flag to determine whether to stop process if pruning is triggered
    EXIT_ON_PRUNING: bool = True
    # Number of samples used to compute running evaluation stats
    example_eval_size: int = 256

    enable_lr_reduction_on_plateau: bool = False
    lr_reduction_plateau_config: LearnRateReductOnPlateauConfig = field(
        default_factory=LearnRateReductOnPlateauConfig)

    # Logging and export
    mlflow_logging: bool = True  # Enable MLFlow logging
    mlflow_experiment_name: str | None = None
    eval_example: bool = False  # Evaluate example input during training
    label_scaling_factors: torch.Tensor | None = None
    checkpoint_dir: str = "./checkpoints"   # Directory to save model checkpoints
    modelName: str = "trained_model"        # Name of the model to be saved
    # Option to enable automatic export to ONNx (attempt)
    export_best_to_onnx: bool = False
    mlflow_unwrap_params_depth: int = 1

    # Optimization parameters
    lr_scheduler: Any | None = None
    initial_lr: float = 1e-4
    optim_momentum: float = 0.75  # Momentum value for SGD optimizer
    optimizer: Any | None = torch.optim.Adam  # optimizer class

    # Model checkpoint load if any
    checkpoint_to_load: str | None = None  # Path to model checkpoint to load
    load_strict: bool = False  # Load model checkpoint with strict matching of parameters
    load_traced: bool = False  # Load model as traced model

    # Hardware/special settings
    # Default device is None at import time
    device: torch.device | str | None = None
    use_torch_amp: bool = False  # Decide whether to use torch AMP for training

    # OPTUNA MODE options
    optuna_trial: Any = None  # Optuna optuna_trial object

    def __post_init__(self):

        if self.device is None:
            self.device = GetDeviceMulti(expected_max_vram_gb=2.0)

        if not(torch.is_tensor(self.label_scaling_factors)) and self.label_scaling_factors is not None:
            # Print warning
            print(
                "\033[38;5;208mWARNING: labels_scaling_factors is not a torch.Tensor. Overriden to None.\033[0m")
            # Set to none
            self.label_scaling_factors = None
        
        if self.label_scaling_factors is None and self.data_augmentation_module is None:
            # Print warning
            print("\033[38;5;208mWARNING: labels_scaling_factors is None and data_augmentation_module is None. No labels scaling will be applied.\033[0m")

        elif isinstance(self.data_augmentation_module, ImageAugmentationsHelper):
            if self.data_augmentation_module.augs_cfg.label_scaling_factors is not None:
                self.label_scaling_factors = torch.Tensor(
                    self.data_augmentation_module.augs_cfg.label_scaling_factors)

        # Compute reciprocal of label_scaling_factors to use multiplication instead of division at runtime
        if self.label_scaling_factors is not None:
            self.label_scaling_factors = 1 / self.label_scaling_factors

        # Switch AMP off if device is CPU
        if self.device == 'cpu' and self.use_torch_amp is True:
            print(
                "\033[38;5;208mWARNING: torch AMP required, but device is CPU. Overriden to False.\033[0m")
            self.use_torch_amp = False

    def __copy__(self, instanceToCopy: 'ModelTrainingManagerConfig') -> 'ModelTrainingManagerConfig':
        """
        Create a shallow copy of the ModelTrainingManagerConfig instance.

        Returns:
            ModelTrainingManagerConfig: A new instance of ModelTrainingManagerConfig with the same configuration.
        """
        return ModelTrainingManagerConfig(**instanceToCopy.get_config_dict())

    # DEVNOTE: dataclass generates __init__() automatically
    # Same goes for __repr()__ for printing and __eq()__ for equality check methods

    def get_config_dict(self) -> dict:
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
    def load_from_yaml(cls, yamlFile: str | IO) -> 'ModelTrainingManagerConfig':
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

    @classmethod  # Why did I defined this class instead of using the __init__ method for dataclasses?
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

        # Check if any required field is missing (those with default values)
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
    def getConfigParamsNames(cls) -> list:
        '''Method to return the names of all parameters in the configuration class'''
        return [f.name for f in fields(cls)]


# TODO: define enum class for optimizers selection if not provided as instance
class enumOptimizerType(Enum):
    SGD = 0
    ADAM = 1
    ADAMW = 2

# %% ModelTrainingManager class - 24-07-2024
# TODO rework trainer NOT to inherit from Config object!


class ModelTrainingManager(ModelTrainingManagerConfig):
    def __init__(self, model: nn.Module | None,
                 lossFcn: nn.Module | CustomLossFcn,
                 config: ModelTrainingManagerConfig | dict | str,
                 optimizer: optim.Optimizer | enumOptimizerType | None = None,
                 dataLoaderIndex: DataloaderIndex | None = None,
                 paramsToLogDict: dict | None = None) -> None:
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
            # This method only copies the attributes present in the dictionary, which may be a subset.
            super().load_from_dict(config)

        elif isinstance(config, ModelTrainingManagerConfig):
            # Initialize ModelTrainingManagerConfig base instance from ModelTrainingManagerConfig instance
            # Call init of parent class for shallow copy
            super().__init__(**config.get_config_dict())
            self.lr_reduction_plateau_config = config.lr_reduction_plateau_config

        # Check that checkpoint_dir exists, if not create it
        if not os.path.isdir(self.checkpoint_dir):
            Warning(
                f"Checkpoint directory {self.checkpoint_dir} does not exist. Creating it...")
            os.makedirs(self.checkpoint_dir)

        # Initialize ModelTrainingManager-specific attributes

        if self.checkpoint_to_load is not None and model is not None:
            # Load model checkpoint
            try:
                print(
                    f"Checkpoint path specified: {self.checkpoint_to_load}. Attempting to load model with strict flag set to {self.load_strict}...")
                model = LoadModel(model, self.checkpoint_to_load,
                                  False, load_strict=self.load_strict)

            except Exception as errMsg:
                # DEVNOTE: here there should be a timer to automatically stop if no input is given for TBD seconds. Use the library for timed input requests?
                print(
                    f"\033[31mModel checkpoint loading failed with error: \n{errMsg}\033[0m")

                user_input = input(
                    "Continue without loading model checkpoint? [Y/n]: ").lower()

                while user_input not in ['y', 'n', 'yes', 'no']:
                    user_input = input(
                        "Please enter a valid input [Y/n]: ").lower()

                if user_input == 'y' or user_input == 'yes':
                    print("Continuing without loading model checkpoint...")
                elif user_input == 'n' or user_input == 'no':
                    print("Exiting program...")
                    sys.exit(0)

        elif self.checkpoint_to_load is not None and model is None:
            # Load model directly
            model = LoadModel(model=None, model_filename=self.checkpoint_to_load,
                              load_as_traced=self.load_traced, load_strict=False)
        elif model is None:
            raise ValueError(
                "Neither model nor model checkpoint path provided. Cannot continue with optimization process.")

        self.model: torch.nn.Module = (model).to(self.device)
        self.bestModel: torch.nn.Module | None = None
        self.loss_fcn: torch.nn.Module = lossFcn

        self.trainingDataloader: torch.utils.data.Dataloader | None = None
        self.validationDataloader: torch.utils.data.Dataloader | None = None
        # For additional evaluation phase if set provided. Validation loader is used if not.
        self.testingDataloader: torch.utils.data.Dataloader | None = None

        self.trainingDataloaderSize: int = 0
        self.current_epoch: int = 0
        self.num_of_updates: int = 0

        self.currentTrainingLoss: float | None = None
        self.currentValidationLoss: float | None = None
        self.currentMlflowRun = mlflow.active_run()  # Returns None if no active run

        if self.mlflow_experiment_name is not None:
            # Update checkpointing directory to split in subfolders
            self.checkpoint_dir = os.path.join(
                self.checkpoint_dir, self.mlflow_experiment_name)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.current_lr: float = self.initial_lr

        self.paramsToLogDict = None
        if paramsToLogDict is not None:
            self.paramsToLogDict = paramsToLogDict

        # OPTUNA parameters
        # DEVNOTE a Function or Callable object may be placed here. Define abstract class for this
        self.optuna_custom_score = None
        if self.optuna_trial is not None:
            self.OPTUNA_MODE = True
        else:
            self.OPTUNA_MODE = False

        # Set kornia transform device
        if self.data_augmentation_module is not None:
            self.data_augmentation_module = self.data_augmentation_module.to(
                self.device)

        # Initialize dataloaders if provided
        if dataLoaderIndex is not None:
            self.setDataloaders(dataLoaderIndex)

        # Handle override of optimizer inherited from ModelTrainingManagerConfig
        # TODO review implementation
        if optimizer is not None:  # Override
            if isinstance(optimizer, optim.Optimizer):
                self._reinstantiate_optimizer(optimizer)
            elif isinstance(optimizer, enumOptimizerType) or issubclass(optimizer, optim.Optimizer):
                self._define_optimizer(optimizer)
            else:
                Warning(
                    'Overriding of optimizer failed. Attempt to use optimizer from ModelTrainingManagerConfig...')

        else:  # Use optimizer from ModelTrainingManagerConfig
            if self.optimizer is not None:
                if isinstance(self.optimizer, optim.Optimizer):
                    self._reinstantiate_optimizer()
                elif isinstance(self.optimizer, enumOptimizerType) or issubclass(self.optimizer, optim.Optimizer):
                    self._define_optimizer(self.optimizer)
            else:
                raise ValueError(
                    'Optimizer must be specified either in the ModelTrainingManagerConfig as torch.optim.Optimizer instance or as an argument in __init__ of this class!')

        # Additional initializations
        if not (os.path.isdir(self.checkpoint_dir)):
            os.mkdir(self.checkpoint_dir)

        # Define reduction on plateau scheduler
        if self.enable_lr_reduction_on_plateau is not None and self.optimizer is not None:
            config = self.lr_reduction_plateau_config.to_scheduler_kwargs()
            self.lr_reduct_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                          **config)

        # Torch AMP objects
        self.grad_scaler = torch.amp.GradScaler(
            device=self.device, enabled=self.use_torch_amp)

    # Instance methods below
    def _define_optimizer(self, optimizer: optim.Optimizer | enumOptimizerType) -> None:
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
        fused_ = True if "cuda" in self.device else False

        if optimizer == enumOptimizerType.SGD or optimizer == torch.optim.SGD:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.initial_lr, momentum=self.optim_momentum)

        elif optimizer == enumOptimizerType.ADAM or optimizer == torch.optim.Adam:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.initial_lr, fused=fused_)

        elif optimizer == enumOptimizerType.ADAMW or optimizer == torch.optim.AdamW:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.initial_lr, fused=fused_)
        else:
            raise ValueError(
                'Optimizer not recognized. Please provide a valid optimizer type or ID from enumOptimizerType enumeration class.')

    def _reinstantiate_optimizer(self, optimizer_override: optim.Optimizer | None = None) -> None:
        """
        Reinstantiates the optimizer with the same hyperparameters but with the current model parameters.
        """
        if optimizer_override is not None:
            self.optimizer = optimizer_override

        if self.model is None:
            raise ValueError(
                'Model is not defined. Cannot reinstantiate optimizer.')
        optim_class = self.optimizer.__class__
        optim_params = self.optimizer.param_groups[0]

        # Determine which keys __init__ actually accepts
        class_signature_ = inspect.signature(optim_class.__init__)
        valid_keys = set(class_signature_.parameters) - {'self', 'params'}

        filtered_params = {k: v for k,
                           v in optim_params.items() if k in valid_keys}

        # Reinstantiate with valid kwargs arguments
        self.optimizer = optim_class(
            self.model.parameters(), **filtered_params)

        if self.lr_scheduler is not None:
            # Reset initial_lr of each group
            base_lr = self.optimizer.param_groups[0]["lr"]
            for pg in self.optimizer.param_groups:
                pg["initial_lr"] = base_lr

            self._reassign_optimizer_to_scheduler(self.lr_scheduler)

    def _reassign_optimizer_to_scheduler(self, lr_scheduler):
        # Add the new optimizer to *this scheduler
        lr_scheduler.optimizer = self.optimizer

        # If *this scheduler wraps other schedulers, recurse into them
        #    SequentialLR uses `schedulers`, other wrappers may use `_schedulers`.
        for attr in ("schedulers", "_schedulers"):
            children_scheduler = getattr(lr_scheduler, attr, None)
            if children_scheduler is not None:
                for subscheduler in children_scheduler:
                    self._reassign_optimizer_to_scheduler(
                        subscheduler)  # Recurse assignment
                break

    def setDataloaders(self, dataloaderIndex: DataloaderIndex) -> None:
        """
        Sets the training and validation dataloaders using the provided DataloaderIndex.

        Args:
            dataloaderIndex (DataloaderIndex): An instance of DataloaderIndex that provides
                                               the training and validation dataloaders.
        """
        if not isinstance(dataloaderIndex, DataloaderIndex):
            raise TypeError(
                f'{colorama.Fore.RED}Expected dataloaderIndex to be of type DataloaderIndex, but found {type(dataloaderIndex)} instead.')

        self.trainingDataloader: DataLoader = dataloaderIndex.TrainingDataLoader
        self.validationDataloader: DataLoader = dataloaderIndex.ValidationDataLoader

        self.trainingDataloaderSize: int = len(self.trainingDataloader)
        self.validationDataloaderSize: int = len(self.validationDataloader)

        if dataloaderIndex.testingDataLoader is not None:
            self.testingDataloader: DataLoader = dataloaderIndex.testingDataLoader

        print(
            f"Training DataLoader size: {self.trainingDataloaderSize}, Validation DataLoader size: {self.validationDataloaderSize}")

    def get_traced_model(self, device=None):
        # TODO make fail safe (i.e. try except print without stopping execution, because this step can fail for a wide variety of reasons)
        if device is None:
            device = self.device

        # Get internal model (best or model)
        model = self.bestModel if (self.bestModel is not None) else self.model

        try:
            raise NotImplementedError('Method not implemented yet.')

        except Exception as e:
            print(
                f"\033[38;5;208mError while tracing model: {e}\033[0m. Model instance will be return as python object instead.")
            return model

    def _print_session_info(self):
        """
        Prints the session information and configuration settings for the model training.

        This method provides a detailed overview of the current training session, including:
        - Task type
        - Model name
        - Device being used
        - Checkpoint directory and file
        - MLflow logging status
        - Number of epochs
        - Trainer mode (OPTUNA or NORMAL)
        - Initial learning rate
        - Optimizer and scheduler details
        - Batch size
        - Keep-best strategy status
        - Validation augmentation status

        If an augmentation pipeline is defined, it also prints the details of each transform in the pipeline,
        including the class name and parameters.

        Returns:
            None
        """

        # Print out session information to check config
        formatted_output = f"""
        SESSION INFO

        - Task Type:                  {self.tasktype}
        - Model Name:                 {self.modelName}
        - Device:                     {self.device}
        - Checkpoint Directory:       {self.checkpoint_dir}
        - Mlflow Logging:             {self.mlflow_logging}
        - Checkpoint file:            {self.checkpoint_to_load}

        SETTINGS

        - Number of Epochs:           {self.num_of_epochs}
        - Trainer Mode:               {'OPTUNA' if self.OPTUNA_MODE else 'NORMAL'}
        - Initial Learning Rate:      {self.initial_lr:0.8g}
        - Optimizer:                  {self.optimizer.__class__.__name__}
        - Scheduler:                  {self.lr_scheduler.__class__.__name__ if self.lr_scheduler is not None else 'None'}
        - Default batch size:         {self.batch_size}
        - Keep-best strategy:         {self.keep_best}
        - Augment validation:         {self.augment_validation_data}
        - Batch accumulation factor:  {self.batch_accumulation_factor}
        """
        print(formatted_output)

        if self.data_augmentation_module is not None and isinstance(self.data_augmentation_module, torch.nn.Sequential):
            print("\033[35mKornia nn.Sequential augmentation pipeline:\033[0m")
            for idx, transform in enumerate(self.data_augmentation_module):
                print(f"  ({idx}): {transform.__class__.__name__}")

                # Print each parameter for the transform
                params = vars(transform)
                for param, value in params.items():
                    print(f"      - {param}: {value}")

        elif isinstance(self.data_augmentation_module, ImageAugmentationsHelper):
            print(
                "\033[35mImage Augmentation helper pipeline configuration:\033[0m")
            pprint.pprint(
                object=self.data_augmentation_module.augs_cfg, indent=2)
        else:
            print("No Kornia augmentation pipeline defined.")

    # def trainInternalModelOneEpoch_(self):
    #    self.trainModelOneEpoch_(model)

    def _train_model_one_epoch(self):
        '''
        Internal method to train the model using the specified datasets and loss function. Not intended to be called as standalone.
        '''

        if self.optimizer is None:
            raise TypeError(
                'Optimizer is not defined. Cannot proceed with training.')

        if self.trainingDataloader is None:
            raise ValueError('No training dataloader provided.')

        running_loss: float = 0.0
        run_time_total: float = 0.0
        loop_iter_number: int = 0
        current_batch: int = 1
        is_last_batch: bool = False

        # Set model instance in training mode and zero out gradients
        self.model.train()
        self.optimizer.zero_grad()

        current_loop_time: float = 0.0
        start_time = time.perf_counter()  # Start timer

        for batch_idx, (X, Y) in enumerate(self.trainingDataloader):

            loop_iter_number += 1

            # Check if batch is the last one
            if batch_idx + 1 == self.trainingDataloaderSize:
                is_last_batch = True

            # Get input and labels and move to target device memory
            # Define input, label pairs for target device
            # DEVNOTE: TBD if this goes here or if to move dataloader to device
            X, Y = X.to(self.device), Y.to(self.device)

            if self.data_augmentation_module is not None:
                if self.enable_augs_autograd:
                    # Perform data augmentation with gradients
                    X, Y = self.augment_data_batch(X, Y)
                else:
                    # Perform data augmentation without gradients
                    with torch.no_grad():
                        X, Y = self.augment_data_batch(X, Y)

            elif self.label_scaling_factors is not None:
                # Apply label scaling factors if provided and no augmentation module is set
                Y = Y * self.label_scaling_factors.to(Y.device)

            # Perform FORWARD PASS to get predictions (autocast is no-ops if use_torch_amp == false)
            device_ = self.device
            if device_ is None:
                raise ValueError('Device is not defined. Cannot proceed with training.')
            elif isinstance(device_, torch.device):
                device_ = device_.type

            with torch.autocast(device_type=device_, dtype=torch.float16, enabled=self.use_torch_amp):
                # Evaluate model at input, calls forward() method
                predicted_values = self.model(X)

                # Evaluate loss function to get loss value dictionary
                train_loss_dict = self.loss_fcn(predicted_values, Y)

                # Get loss value from dictionary
                train_loss_value = train_loss_dict.get('lossValue') if isinstance(
                    train_loss_dict, dict) else train_loss_dict

                assert train_loss_value is not None, "Loss value cannot be None. Check loss function implementation."

                if self.batch_accumulation_factor > 1:
                    train_loss_value /= self.batch_accumulation_factor

            # TODO: here one may log intermediate metrics at each update
            # if self.mlflow_logging:
            #     mlflow.log_metrics()

            # Update running value of loss for status bar at current epoch
            running_loss += train_loss_value.detach().cpu().float()

            # Perform BACKWARD PASS to update parameters
            # Compute gradients (wrapped in grad scaler for AMP)
            self.grad_scaler.scale(train_loss_value).backward()

            if ((batch_idx + 1) % self.batch_accumulation_factor == 0) or is_last_batch:
                # Make optimization step and reset gradients
                # Apply gradients from the loss
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()  # Update grad scaler for AMP

                self.optimizer.zero_grad()  # Reset gradients for next iteration
                self.num_of_updates += 1
            else:
                continue  # Accumulate more batches before update

            if self.device is not None:
                if isinstance(self.device, str):
                    if self.device.startswith('cuda'):
                        # Synchronize CUDA stream once here
                        torch.cuda.synchronize()
                elif isinstance(self.device, torch.device):
                    if self.device.type == 'cuda':
                        # Synchronize CUDA stream once here
                        torch.cuda.synchronize()

            # Update total loop time
            current_loop_time = time.perf_counter() - start_time
            run_time_total += current_loop_time

            # Calculate progress
            current_batch = batch_idx + 1
            progress_info = f"  E[{self.current_epoch+1}/{self.num_of_epochs}] - Training: Batch {batch_idx+1}/{self.trainingDataloaderSize}, avg. loss: {running_loss / current_batch:4.5g}, num. of steps: {self.num_of_updates}, single loop time: {1000 * current_loop_time:4.4g} [ms], per-batch avg. time: {1000*run_time_total/loop_iter_number:4.4g} [ms], current lr: {self.current_lr:.06g}"

            # Print progress on the same line
            print(progress_info, end="\r")

            # Reset timer
            start_time = time.perf_counter()

            # TODO: implement management of SWA model
            # if swa_model is not None and epochID >= swa_start_epoch:

        print('\n')  # Add newline after progress bar
        return running_loss / current_batch

    def validateInternalModel_(self):
        """Wrapper function to call validation method on internal model state"""

        validation_loss_value = self.validateModel_(self.model)

        return validation_loss_value

    def validateModel_(self, model: torch.nn.Module):
        """Method to validate the model using the specified datasets and loss function. Not intended to be called as standalone."""
        if self.validationDataloader is None:
            raise ValueError('No validation dataloader provided.')
        if self.model is None:
            raise ValueError('No model provided.')

        # TODO improve this method, no real need to have two separate cycles for classification and regression. Just move different code to submethods. Moreover, need to adapt for other applications

        model.eval()
        validation_loss_value = 0.0  # Accumulation variables
        # batchMaxLoss = 0
        # validationData = {}  # Dictionary to store validation data

        # Backup the original batch size (DEVNOTE TODO Does it make sense?)
        original_dataloader = self.validationDataloader

        # Temporarily initialize a new dataloader for validation
        # TODO replace this heuristics with something more grounded and memory aware!
        newBatchSizeTmp = 2 * self.validationDataloader.batch_size

        device_is_cuda = False
        if self.device is not None:
            if isinstance(self.device, str):
                if self.device.startswith('cuda'):
                    device_is_cuda = True
            elif isinstance(self.device, torch.device):
                if self.device.type == 'cuda':
                    device_is_cuda = True

        tmpDataloader = DataLoader(original_dataloader.dataset,
                                   batch_size=newBatchSizeTmp,
                                   shuffle=False,
                                   drop_last=False,
                                   pin_memory=device_is_cuda,
                                   num_workers=0
                                   )

        num_of_batches = len(tmpDataloader)
        dataset_size = len(tmpDataloader.dataset)

        with torch.no_grad():
            run_time_total = 0.0

            if self.tasktype == TaskType.CLASSIFICATION:
                # TODO rework trainer structure entirely, quite old and outdated now

                if not (isinstance(self.loss_fcn, torch.nn.CrossEntropyLoss)):
                    raise NotImplementedError(
                        'Current classification validation function only supports nn.CrossEntropyLoss.')

                correct_predictions = 0

                for batch_idx, (X, Y) in enumerate(tmpDataloader):

                    # Start timer for batch processing time
                    start_time = time.perf_counter()

                    # Get input and labels and move to target device memory
                    X, Y = X.to(self.device), Y.to(self.device)

                    # Run data agmentations
                    if self.data_augmentation_module is not None and self.augment_validation_data:
                        # DEVNOTE current implementation limited to keypoints.
                        # How to allow extraction of entries in Y?
                        X, Y = self.augment_data_batch(X, Y)
                    else:
                        # Use internal scaling factors to scale labels
                        assert self.label_scaling_factors is not None, 'Labels scaling factors are not defined. Cannot scale labels.'
                        Y = Y * self.label_scaling_factors.to(Y.device)

                    # Perform FORWARD PASS
                    predicted_val = model(X)  # Evaluate model at input

                    # Evaluate loss function to get loss value dictionary
                    valid_loss_dict = self.loss_fcn(predicted_val, Y)
                    validation_loss_value += valid_loss_dict.get('lossValue') if isinstance(
                        valid_loss_dict, dict) else valid_loss_dict.item()

                    # Evaluate how many correct predictions (assuming CrossEntropyLoss)
                    correct_predictions += (predicted_val.argmax(1)
                                            == Y).type(torch.float).sum().item()

                    # Accumulate batch processing time
                    run_time_total += time.perf_counter() - start_time

                    # Calculate progress
                    current_batch = batch_idx + 1
                    progress_info = f"  E[{self.current_epoch+1}/{self.num_of_epochs}] - Validation: Batch {batch_idx+1}/{num_of_batches}, avg. loss: {validation_loss_value / current_batch:4.5g}, per-batch avg. time: {1000*run_time_total/(current_batch):4.4g} [ms]"

                    # Print progress on the same line
                    print(progress_info, end="\r")

                # Compute batch size normalized loss value
                validation_loss_value /= num_of_batches
                # Compute percentage of correct classifications over dataset size
                correct_predictions /= dataset_size
                print(
                    f"\n\t\tFinal score - accuracy: {(100*correct_predictions):0.2f}%, average loss: {validation_loss_value:.4f}\n")
                return validation_loss_value, correct_predictions

            elif self.tasktype == TaskType.REGRESSION:

                for batch_idx, (X, Y) in enumerate(tmpDataloader):

                    # Start timer for batch processing time
                    start_time = time.perf_counter()

                    # Get input and labels and move to target device memory
                    X, Y = X.to(self.device), Y.to(self.device)

                    # Perform data augmentation on batch
                    if self.data_augmentation_module is not None and self.augment_validation_data:
                        X, Y = self.augment_data_batch(X, Y)
                    else:
                        # Use internal scaling factors to scale labels
                        assert self.label_scaling_factors is not None, 'Labels scaling factors are not defined. Cannot scale labels.'
                        Y = Y * self.label_scaling_factors.to(Y.device)
                    
                    # Perform FORWARD PASS
                    predicted_val = model(X)  # Evaluate model at input

                    # Evaluate loss function to get loss value dictionary
                    valid_loss_dict = self.loss_fcn(predicted_val, Y)

                    # Get loss value from dictionary
                    validation_loss_value += valid_loss_dict.get('lossValue') if isinstance(
                        valid_loss_dict, dict) else valid_loss_dict.item()

                    # Accumulate batch processing time
                    run_time_total += time.perf_counter() - start_time

                    # Calculate progress
                    current_batch = batch_idx + 1
                    progress = f"\tValidation: Batch {batch_idx+1}/{num_of_batches}, average loss: {validation_loss_value / current_batch:4.5g}, average loop time: {1000 * run_time_total/(current_batch):4.2f} [ms]"

                    # Print progress on the same line
                    sys.stdout.write('\r' + progress)
                    sys.stdout.flush()

                # Compute batch size normalized loss value
                validation_loss_value /= num_of_batches
                print(
                    f"\n\t\tFinal score - avg. loss: {validation_loss_value:4.5g}\n")

                return validation_loss_value

            else:
                raise NotImplementedError(
                    'Custom task type not implemented yet.')

    def trainAndValidate(self):
        """
        trainAndValidate _summary_

        _extended_summary_

        :raises optuna.TrialPruned: _description_
        :raises optuna.TrialPruned: _description_
        :raises optuna.TrialPruned: _description_
        :raises optuna.TrialPruned: _description_
        :return: _description_
        :rtype: _type_
        """
        colorama.init(autoreset=True)

        if self.trainingDataloader is None:
            raise ValueError(
                f'{colorama.Fore.RED}No training dataloader provided. Cannot proceed.')

        if self.validationDataloader is None:
            raise ValueError(
                f'{colorama.Fore.RED}No validation dataloader provided or split from training. Cannot proceed.')

        # Spin mlflow pipeline if required
        self.startMlflowRun()  # DEVNOTE: TODO split into more subfunctions

        print(f'\n\n{colorama.Style.BRIGHT}{colorama.Fore.BLUE}-------------------------- Training and validation session start --------------------------\n')
        self._print_session_info()

        model_save_name = None
        no_new_best_counter = 0

        try:
            if self.OPTUNA_MODE:
                trial_printout = f" of trial {self.optuna_trial.number}"
            else:
                trial_printout = ""

            ### Pre-training operations
            if self.checkpoint_to_load is not None:
                # If restart from check point, perform validation first to set current best
                validation_loss_value = self.validateModel_(self.model)

                if isinstance(validation_loss_value, tuple):
                    validation_loss_value = validation_loss_value[0]

                self.currentValidationLoss: float = validation_loss_value
                self.bestValidationLoss: float = validation_loss_value
                self.bestModel: torch.nn.Module | None = copy.deepcopy(self.model).to('cpu')

            ########################################
            ### Loop over epochs
            for epoch_num in range(self.num_of_epochs):

                print(f"\n{colorama.Style.BRIGHT}{colorama.Fore.GREEN}Training epoch" + trial_printout,
                      f"{colorama.Style.BRIGHT}{colorama.Fore.GREEN}: {epoch_num+1}/{self.num_of_epochs}")
                cycle_start_time = time.time()

                # Get current learning rate
                self.current_lr = self.optimizer.param_groups[0]['lr']

                # Log basic data
                if self.mlflow_logging:
                    mlflow.log_metric('lr', self.current_lr,
                                      step=self.current_epoch)

                #####################################
                ### Perform training for one epoch
                tmp_train_loss = self._train_model_one_epoch()

                if isinstance(tmp_train_loss, torch.Tensor):
                    tmp_train_loss = tmp_train_loss.item()
                    
                # Check if train loss is NaN or Inf
                if isnan(tmp_train_loss) or math.isinf(tmp_train_loss):
                    raise ValueError(
                        f"Detected training failure. Training loss is {'NaN' if isnan(tmp_train_loss) else 'Inf'}. Run stop.")

                # Perform validation at current epoch
                tmp_valid_loss = self.validateInternalModel_()

                # TODO clarify intent of this, remove if not really necessary
                if isinstance(tmp_valid_loss, tuple):
                    tmp_valid_loss = tmp_valid_loss[0]

                if isinstance(tmp_valid_loss, torch.Tensor):
                    tmp_valid_loss = tmp_valid_loss.item()
                
                # Check if validation loss is NaN or Inf
                if isnan(tmp_valid_loss) or math.isinf(tmp_valid_loss):
                    raise ValueError(
                        f"Detected training failure. Validation loss is {'NaN' if isnan(tmp_valid_loss) else 'Inf'}. Run stop.")
                    
                # At epoch 0, set initial validation loss
                if self.currentValidationLoss is None:
                    self.currentValidationLoss: float = tmp_valid_loss
                    self.bestValidationLoss: float = tmp_valid_loss

                ######################################
                ### Post training-validation operations

                # Optuna functionalities
                # Report validation loss to Optuna pruner
                if self.OPTUNA_MODE == True:
                    # Compute average between training and validation loss // TODO: verify feasibility of using the same obj function as sampler
                    if self.optuna_custom_score is not None:
                        # optuna_loss = self.optuna_custom_score()
                        raise NotImplementedError(
                            'Branch of code still in development.')
                    else:
                        optuna_loss = (tmp_train_loss + tmp_valid_loss) / 2

                    self.optuna_trial.report(optuna_loss, step=epoch_num)

                    if self.optuna_trial.should_prune():
                        raise optuna.TrialPruned()
                else:
                    # Execute post-epoch operations
                    self.evalExample(self.example_eval_size)

                # Update stats if new best model found (independently of keep_best flag)
                if tmp_valid_loss <= self.bestValidationLoss:
                    self.bestEpoch = epoch_num
                    self.bestValidationLoss = tmp_valid_loss
                    no_new_best_counter = 0
                else:
                    no_new_best_counter += 1

                # "Keep best" strategy implementation (trainer will output the best overall model at cycle end)
                # DEVNOTE: this could go into a separate method
                if self.keep_best:
                    if tmp_valid_loss <= self.bestValidationLoss:

                        # Transfer best model to CPU to avoid additional memory allocation on GPU
                        self.bestModel: torch.nn.Module | None = copy.deepcopy(
                            self.model).to('cpu')

                        # Delete previous best model checkpoint if it exists
                        if model_save_name is not None:

                            # Get file name with modelSaveName as prefix
                            checkpoint_files = glob.glob(
                                f"{os.path.join(self.checkpoint_dir, self.modelName)}_epoch*")

                            if checkpoint_files:
                                # If multiple files match, delete all or choose one (e.g., the first one)
                                for file in checkpoint_files:
                                    if os.path.exists(file):
                                        os.remove(file)
                                        break

                        # Save temporary best model
                        model_save_name = os.path.join(
                            self.checkpoint_dir, self.modelName + f"_epoch_{self.bestEpoch}")

                        if self.bestModel is not None:
                            SaveModel(model=self.bestModel, model_filename=model_save_name,
                                      save_mode=AutoForgeModuleSaveMode.MODEL_ARCH_STATE,
                                      target_device='cpu')

                # Update current training and validation loss values
                self.currentTrainingLoss: float = tmp_train_loss
                self.currentValidationLoss: float = tmp_valid_loss

                if self.mlflow_logging and self.currentMlflowRun is not None:
                    if self.currentTrainingLoss is not None:
                        mlflow.log_metric(
                            'train_loss', self.currentTrainingLoss, step=self.current_epoch)

                    if self.currentValidationLoss is not None:
                        mlflow.log_metric(
                            'validation_loss', self.currentValidationLoss, step=self.current_epoch)

                    mlflow.log_metric(
                        'best_validation_loss', self.bestValidationLoss, step=self.current_epoch)
                    mlflow.log_metric(
                        'num_of_updates', self.num_of_updates, step=self.current_epoch)

                print('\tCurrent best at epoch {best_epoch}, with validation loss: {best_loss:.06g}'.format(
                    best_epoch=self.bestEpoch, best_loss=self.bestValidationLoss))
                print(
                    f'\tEpoch cycle duration: {((time.time() - cycle_start_time) / 60):.4f} [min]')

                # "Early stopping" strategy implementation
                if self.OPTUNA_MODE == False:
                    if self.checkForEarlyStop(no_new_best_counter):
                        break
                elif self.OPTUNA_MODE == True:
                    # Safety exit for model divergence
                    if tmp_train_loss >= 1E8 or tmp_valid_loss >= 1E8:
                        raise optuna.TrialPruned()

                # Post epoch operations
                self.updateLearningRate_()  # Update learning rate if scheduler is provided
                self.current_epoch += 1
            # END OF TRAINING-VALIDATION LOOP

            # Model saving code
            if model_save_name is not None:
                if os.path.exists(model_save_name):
                    os.remove(model_save_name)

            if self.keep_best:
                print('Best model saved from epoch: {best_epoch} with validation loss: {best_loss:.4f}'.format(
                    best_epoch=self.bestEpoch, best_loss=self.bestValidationLoss))

            with torch.no_grad():
                examplePair = next(iter(self.validationDataloader))
                model_save_name = os.path.join(
                    self.checkpoint_dir, self.modelName + f"_epoch_{self.bestEpoch}")

                if self.bestModel is not None:
                    SaveModel(model=self.bestModel, model_filename=model_save_name,
                              save_mode=AutoForgeModuleSaveMode.MODEL_ARCH_STATE,
                              example_input=examplePair[0],
                              target_device=self.device)
                else:
                    print(
                        "\033[38;5;208mWARNING: SaveModel skipped due to bestModel being None!\033[0m")

            if self.mlflow_logging:
                mlflow.log_param('checkpoint_best_epoch', self.bestEpoch)

            # Post-training operations
            print('Training and validation cycle completed.')
            if self.mlflow_logging:
                mlflow.end_run(status='FINISHED')

            return self.bestModel if self.keep_best else self.model

        except KeyboardInterrupt:
            print(
                '\n\033[38;5;208mModelTrainingManager stopped execution due to KeyboardInterrupt. Run marked as KILLED.\033[0m')

            # Save best model up to current epoch if not None
            try:
                # TODO replace by a trainer export method
                if self.bestModel is not None:
                    examplePair = next(iter(self.validationDataloader))
                    model_save_name = os.path.join(
                        self.checkpoint_dir, self.modelName + f"_epoch_{self.bestEpoch}")

                    if self.bestModel is not None:
                        SaveModel(model=self.bestModel, model_filename=model_save_name,
                                  save_mode=AutoForgeModuleSaveMode.MODEL_ARCH_STATE,
                                  example_input=examplePair[0],
                                  target_device=self.device)

                        # TODO
                        try:
                            if self.export_best_to_onnx:
                                pass
                        except:
                            pass

                    print(
                        f"\t\033[38;5;208mBest model checkpoint saved correctly.\033[0m")
            except Exception as err:
                print(
                    f"\t\033[31mAttempt to save best model checkpoint failed due to: {str(err)}\033[0m")

            if self.mlflow_logging:
                mlflow.end_run(status='KILLED')  # Mark run as killed

            if self.OPTUNA_MODE:
                # signal.signal(signal.SIGALRM, _timeout_handler) # Does not work for Windows
                while True:
                    try:
                        user_input = inputimeout(
                            '\n\n\033[38;5;208mStop execution (Y) or mark as pruned (N)?\033[0m',
                            timeout=60
                        ).strip().lower()

                        if user_input in ('n', 'no'):
                            raise optuna.TrialPruned()

                        elif user_input in ('y', 'yes'):
                            # exit the loop & program gracefully
                            sys.exit(0)
                        else:
                            print(
                                "Invalid choice — please type Y or N (timeout set to 60 s).")

                    except TimeoutOccurred:
                        print(
                            "\033[31mTimeout error triggered, program stop.\033[0m")

                    except EOFError:
                        sys.exit("No input available, program stop.")

            # Exit from program gracefully
            if self.EXIT_ON_PRUNING:
                sys.exit(0)
            else:
                # Ask user what to do
                while True:
                    try:
                        user_input = inputimeout(
                            '\n\n\033[38;5;208mKeyboard interrupt pruning. Stop execution of program (Y) or continue (N)?\033[0m',
                            timeout=60
                        ).strip().lower()

                        if user_input in ('n', 'no'):
                            print('\033[33mContinuing execution...\033[0m')
                            break  # Exit the loop and continue execution

                        elif user_input in ('y', 'yes'):
                            sys.exit(0)  # Exit the program gracefully
                        else:
                            print("Invalid choice — please type Y or N (timeout set to 60 s).")

                    except TimeoutOccurred:
                        print("\033[31mTimeout error triggered, program stop.\033[0m")
                        sys.exit(0)

                    except EOFError:
                        sys.exit("No input available, program stop.")

        except optuna.TrialPruned:
            # Optuna trial kill raised
            print(
                '\033[33m\nModelTrainingManager stopped execution due to Optuna Pruning signal. Run marked as KILLED.\033[0m')

            if self.mlflow_logging:
                mlflow.end_run(status='KILLED')
            # Re-raise exception to stop optuna trial --> this is required due to how optuna handles it.
            raise optuna.TrialPruned()

        except Exception as e:  # Any other exception
            max_chars = 2000  # Define the max length you want to print
            error_message = str(e)[:max_chars]

            traceback_limit = 8
            traceback_ = traceback.format_exc(limit=traceback_limit)

            print(f"\033[31m\nError during training and validation cycle: {error_message}...\nTraceback includes most recent {traceback_limit} calls. {traceback_}\033[0m")

            if self.mlflow_logging:
                mlflow.end_run(status='FAILED')

            # Exit from program gracefully
            sys.exit(0)

    def augment_data_batch(self, *raw_inputs: torch.Tensor):

        if self.data_augmentation_module is None:
            print(f"{colorama.Fore.LIGHTRED_EX}WARNING: augment_data_batch was called, but data_augmentation_module is None. No transformation was applied.")
            return raw_inputs

        # Apply ImageAugmentationsHelper forward method
        aug_inputs = self.data_augmentation_module(*raw_inputs)

        return aug_inputs

    def evalExample(self, num_samples: int = 128) -> None:
        # TODO Extend method distinguishing between regression and classification tasks

        if self.model is None:
            raise ValueError(
                f'{colorama.Fore.RED}No model provided. Cannot proceed with inference!')

        self.model.eval()
        try:
            if self.eval_example:
                # exampleInput = GetSamplesFromDataset(self.validationDataloader, 1)[0][0].reshape(1, -1)
                # if self.mlflow_logging: # TBC, not sure it is useful
                #    # Log example input to mlflow
                #    mlflow.log_???('example_input', exampleInput)

                with torch.no_grad():
                    average_loss = 0.0
                    num_of_batches = 0
                    samples_counter = 0

                    average_prediction = None
                    worst_prediction_err = None
                    prediction_errors = None
                    correct_predictions = 0

                    # Define temporary dataloader
                    if self.testingDataloader is not None:
                        tmp_loader = self.testingDataloader
                    else:
                        tmp_loader = self.validationDataloader

                    eval_dataloader = DataLoader(
                        dataset=tmp_loader.dataset,
                        batch_size=tmp_loader.batch_size,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=False,
                        num_workers=0
                    )

                    if eval_dataloader is None:
                        print(f'{colorama.Fore.RED}Evaluation dataloader is None. Cannot proceed in testing the model. Manager will continue with training and validation.')
                        return

                    label_scaling_factors = 1.0

                    if self.label_scaling_factors is not None:
                        # Use internal scaling factors to scale labels
                        label_scaling_factors = 1 / self.label_scaling_factors.to(self.device)
                        
                    while samples_counter < num_samples:
                        # Note that this returns a batch of size given by the dataloader
                        examplePair = next(iter(eval_dataloader))

                        # TODO modify for tuple input support
                        X = examplePair[0].to(self.device)
                        Y = examplePair[1].to(self.device)

                        if self.data_augmentation_module is not None and self.augment_validation_data:
                            X, Y = self.augment_data_batch(X, Y)
                        else:
                            # Use internal scaling factors to scale labels
                            assert self.label_scaling_factors is not None, 'Labels scaling factors are not defined. Cannot scale labels.'
                            Y = Y * self.label_scaling_factors.to(self.device)

                        # Perform FORWARD PASS
                        example_predictions = self.model(
                            X)  # Evaluate model at input

                        try:
                            if example_predictions.shape != Y.shape:
                                # Attempt to match shapes
                                Y = Y[:, 0:example_predictions.size(1)]
                        except:
                            print(f'{colorama.Fore.RED}Predicted value shape {example_predictions.shape} did not match labels shape {Y.shape} and automatic matching attempt failed. Evaluation stopped.')
                            return

                        # Task specific code
                        if not self.tasktype == TaskType.REGRESSION and not self.tasktype == TaskType.CLASSIFICATION:
                            print(
                                f'{colorama.Fore.RED}Invalid Task type. Cannot proceed in evaluation. Continuing execution...')
                            return

                        if self.tasktype == TaskType.REGRESSION:
                            if prediction_errors is None:
                                # Initialize predictions errors tensor
                                prediction_errors = example_predictions - Y
                            else:
                                # Concatenate if existing, for later stats computation
                                prediction_errors = torch.cat(
                                    [prediction_errors, example_predictions - Y], dim=0)

                            # Compute loss for each input separately
                            # DEVNOTE must return a Tensor!
                            output_loss = self.loss_fcn(example_predictions, Y)

                            # Compute running average of loss
                            average_loss += output_loss.item()

                        elif self.tasktype == TaskType.CLASSIFICATION:

                            if not (isinstance(self.loss_fcn, torch.nn.CrossEntropyLoss)):
                                raise NotImplementedError(
                                    'Current classification validation function only supports nn.CrossEntropyLoss.')

                            example_loss_output = self.loss_fcn(
                                example_predictions, Y)

                            average_loss += example_loss_output.get('lossValue') if isinstance(
                                # This assumes a standard format of the output dictionary from custom loss
                                example_loss_output, dict) else example_loss_output.item()

                            # Evaluate how many correct predictions (assuming CrossEntropyLoss)
                            correct_predictions += (example_predictions.argmax(1)
                                                    == Y).type(torch.float).sum().item()

                        # Count samples and batches
                        samples_counter += X.size(0)
                        num_of_batches += 1

                    # Post evaluation stats computation
                    if self.tasktype == TaskType.REGRESSION:

                        # Compute average prediction over all samples
                        average_prediction = label_scaling_factors * \
                            torch.mean(prediction_errors, dim=0)
                        average_loss /= num_of_batches

                        # Get worst prediction error over all samples
                        worst_prediction_err, _ = torch.max(
                            torch.abs(prediction_errors), dim=0)
                        worst_prediction_err *= label_scaling_factors

                        # Get median prediction error over all samples
                        median_prediction_err, _ = torch.median(
                            torch.abs(prediction_errors), dim=0)
                        median_prediction_err *= label_scaling_factors

                        quantile95_prediction_err = torch.quantile(
                            torch.abs(prediction_errors), 0.95, 0)
                        quantile95_prediction_err *= label_scaling_factors

                        quantile99_prediction_err = torch.quantile(
                            torch.abs(prediction_errors), 0.99, 0)
                        quantile99_prediction_err *= label_scaling_factors

                        # TODO (TBC): log example in mlflow?
                        # if self.mlflow_logging:
                        #    print('TBC')

                        print(f"{colorama.Style.BRIGHT}{colorama.Fore.YELLOW}\tSample error statistics on batch of {num_samples} samples:",
                            "\n\tCorresponding average loss: ", average_loss, f"{colorama.Style.RESET_ALL}")

                        print(
                            f"\n\tAverage prediction (scaled) errors per component: \n\t\t", average_prediction)
                        print(
                            f"\n\tWorst abs. prediction (scaled) errors per component: \n\t\t", worst_prediction_err)
                        print(f"\n\tQuantile 99 abs. prediction (scaled) errors per component: \n\t\t",
                            quantile99_prediction_err)
                        print(f"\n\tQuantile 95 abs. prediction (scaled) errors per component: \n\t\t",
                            quantile95_prediction_err)
                        print(
                            f"\n\tMedian abs. prediction (scaled) errors per component: \n\t\t", median_prediction_err)
                        print("\n")

                    elif self.tasktype == TaskType.CLASSIFICATION:

                        average_loss /= num_of_batches  # Compute batch size normalized loss value

                        # Compute percentage of correct classifications over dataset size
                        correct_predictions /= samples_counter
                        print(f"\n\tExample prediction with {samples_counter} samples: Classification accuracy:",
                            f"{(100*correct_predictions):>0.2f} % , average loss: {average_loss:>4f}\n")

                    else:
                        raise TypeError('Invalid Task type.')
        except Exception as err:
            print(f"{colorama.Fore.RED}Error during example evaluation: {err}. Continuing execution...")

    def updateLearningRate_(self):
        """
        Updates the learning rate of the optimizer if a learning rate scheduler is provided.
        """
        if self.lr_scheduler is not None and self.optimizer is not None:
            # Perform step of learning rate scheduler if provided
            self.optimizer.zero_grad()  # Reset gradients for safety
            self.lr_scheduler.step()

            # Get learning rate value after step
            new_lr = self.lr_scheduler.get_last_lr()[0]

            if self.enable_lr_reduction_on_plateau and self.currentValidationLoss is not None:
                self.lr_reduct_plateau.step(
                    metrics=self.currentValidationLoss)  # Verify if on plateau
                new_lrs = [g['lr'] for g in self.optimizer.param_groups]
                has_reduced_on_plateau = any(
                    lr < self.current_lr for lr in new_lrs)

                if has_reduced_on_plateau:
                    # Get the minimum learning rate after reduction
                    new_lr = min(new_lrs)
            else:
                has_reduced_on_plateau = False

            # FIXME: it seems that step does not work ok if cosine annealing is working

            # Print info
            if has_reduced_on_plateau:

                print('\n{light_blue}Learning rate reduced on plateau: {prev_lr:.6g} --> {new_lr:.6g} for {num_cooldown_epochs} epochs {reset}\n'.format(light_blue=colorama.Fore.LIGHTRED_EX,
                                                                                                                                                         prev_lr=self.current_lr,
                                                                                                                                                         new_lr=new_lr,
                                                                                                                                                         num_cooldown_epochs=self.lr_reduction_plateau_config.num_cooldown_epochs,
                                                                                                                                                         reset=colorama.Style.RESET_ALL))

            else:
                print('\n{light_blue}Learning rate changed: {prev_lr:.6g} --> {new_lr:.6g}{reset}\n'.format(light_blue=colorama.Fore.LIGHTBLUE_EX,
                                                                                                            prev_lr=self.current_lr,
                                                                                                            new_lr=new_lr,
                                                                                                            reset=colorama.Style.RESET_ALL))

            # Save the new learning rate
            self.current_lr = new_lr

    def checkForEarlyStop(self, counter: int) -> bool:
        """
        Checks if the early stopping criteria have been met.
        Parameters:
        counter (int): The current count of epochs or iterations without improvement.
        Returns:
        bool: True if early stopping criteria are met and training should stop, False otherwise.
        """
        returnValue = False

        if self.enable_early_pruning:
            if counter >= self.pruning_patience:
                print(
                    '\033[38;5;208mEarly stopping criteria met: ModelTrainingManager execution stop. Run marked as KILLED.\033[0m')
                returnValue = True
                if self.mlflow_logging:
                    mlflow.end_run(status='KILLED')

        return returnValue

    def startMlflowRun(self):
        """
        Starts a new MLflow run if MLflow logging is enabled.

        If there is an active MLflow run, it ends the current run before starting a new one.
        Updates the current MLflow run to the newly started run.

        Args:
            None

        Raises:
            Warning: If MLflow logging is disabled.

        Prints:
            Messages indicating the status of the MLflow runs.
        """
        if self.mlflow_logging:
            if self.model is None:
                raise ValueError('No model provided for MLflow logging.')

            if self.currentMlflowRun is not None:
                mlflow.end_run()
                print(('\033[38;5;208m\nActive mlflow run {active_run} ended before creating new one.\033[0m').format(
                    active_run=self.currentMlflowRun.info.run_name))

            mlflow.start_run()
            # Update current mlflow run
            self.currentMlflowRun = mlflow.active_run()
            print(colorama.Fore.BLUE + ('\nStarted new Mlflow run with name: {active_run}.').format(
                active_run=self.currentMlflowRun.info.run_name) + colorama.Style.RESET_ALL)

            # Set model name from mlflow run
            self.modelName = self.currentMlflowRun.info.run_name

            try:
                # Log configuration parameters
                ModelTrainerConfigParamsNames = ModelTrainingManagerConfig.getConfigParamsNames()
                RecursiveLogParamsInDict({key: getattr(self, key)
                                        for key in ModelTrainerConfigParamsNames}, self.mlflow_unwrap_params_depth)

                # Log model info (size, number of parameters)
                mlflow.log_param('num_trainable_parameters', sum(p.numel()
                                for p in self.model.parameters() if p.requires_grad))

                mlflow.log_param('num_total_parameters', sum(p.numel()
                                for p in self.model.parameters()))

                size_mb = ComputeModelParamsStorageSize(self.model)
                mlflow.log_param(key='model_storage_MB', value=round(size_mb, 4))

                # Log additional parameters if provided
                if self.paramsToLogDict is not None:
                    # TODO improve logging to prevent duplicated key. Current workaround is to make it fail withing the function
                    RecursiveLogParamsInDict(self.paramsToLogDict, self.mlflow_unwrap_params_depth)
                    
            except Exception as e:

                traceback_limit = 10
                traceback_ = traceback.format_exc(limit=traceback_limit)

                print(colorama.Fore.RED + f"Run failed. Error logging parameters: {e}.\nTraceback of calls: {traceback_}" + colorama.Style.RESET_ALL)
                mlflow.end_run(status='FAILED')
                sys.exit(1)

            if self.OPTUNA_MODE:
                mlflow.log_param('optuna_trial_ID', self.optuna_trial.number)
                self.optuna_trial.set_user_attr('mlflow_name', self.modelName)
                self.optuna_trial.set_user_attr('mlflow_ID', self.currentMlflowRun.info.run_id)

    def exportModel(self):
        pass
    # TODO move code to save models here, with tracing/onnx option (fail-safe: save pth if cannot save traced or equivalence fails)

# %% Function to freeze a generic nn.Module model parameters to avoid backpropagation


def FreezeModel(model: nn.Module) -> nn.Module:
    model.requires_grad_(False)
    return model

####################################################################################################

# LEGACY (no longer maintained) FUNCTIONS - 18/09/2024
# %% Function to perform one step of training of a model using dataset and specified loss function - 04-05-2024
# Updated by PC 04-06-2024

# TODO Update function to resemble trainer "train one epoch", useful for experiments


def TrainModel(dataloader: DataLoader,
               model: nn.Module,
               lossFcn: nn.Module,
               optimizer,
               epochID: int,
               device=None,
               lr_scheduler=None,
               swa_scheduler=None,
               swa_model=None,
               swa_start_epoch: int = 15) -> float | int:
    '''Function to perform one step of training of a model using input dataloader and loss function'''
    model.train()  # Set model instance in training mode ("informing" backend that the training is going to start)

    if device is None:
        device = GetDeviceMulti()

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
            #     print("\t",", ".join([f"{key}: {trainLossOut[key]:.4f}" for key in keys]))    # Update learning rate if scheduler is provided

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


def ValidateModel(dataloader: DataLoader,
                  model: nn.Module,
                  lossFcn: nn.Module,
                  device=None,
                  taskType: str = 'classification') -> float | dict:
    '''Function to validate model using dataset and specified loss function'''
    # Get size of dataset (How many samples are in the dataset)
    size = len(dataloader.dataset)

    if device is None:
        device = GetDeviceMulti()

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
        raise NotImplementedError(
            "This is a deprecated function, please use ModelTrainingManager.")

    with torch.no_grad():  # Tell torch that gradients are not required

        # Backup the original batch size
        original_dataloader = dataloader
        original_batch_size = dataloader.batch_size

        # Temporarily initialize a new dataloader for validation
        # If device is CUDA check for memory availability
        if device.startswith('cuda'):
            allocMem = torch.cuda.memory_allocated(0)
            freeMem = torch.cuda.get_device_properties(
                0).total_memory - torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            estimated_memory_per_sample = allocMem / original_batch_size
            newBatchSizeTmp = min(
                round(0.5 * freeMem / estimated_memory_per_sample), 2048)
        else:
            newBatchSizeTmp = 2 * original_batch_size

        dataloader = DataLoader(
            dataloader.dataset,
            batch_size=newBatchSizeTmp,
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
                correctOuputs += (predVal.argmax(1) ==
                                  Y).type(torch.float).sum().item()

    # Log additional metrics to MLFlow if any
    if lossTerms != {}:
        lossTerms = {key: (value / numberOfBatches)
                     for key, value in lossTerms.items()}
        mlflow.log_metrics(lossTerms, step=0)

    # Restore the original batch size
    dataloader = original_dataloader

    if taskType.lower() == 'classification':
        validationLoss /= numberOfBatches  # Compute batch size normalized loss value
        correctOuputs /= size  # Compute percentage of correct classifications over batch size
        print(
            f"\n VALIDATION (Classification) Accuracy: {(100*correctOuputs):>0.2f}%, Avg loss: {validationLoss:>8f} \n")

    elif taskType.lower() == 'regression':
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


# %% TRAINING and VALIDATION template function (LEGACY, no longer maintained) - 04-06-2024
# @deprecated() # DEVNOTE requires Python 3.13
def TrainAndValidateModel(dataloaderIndex: DataloaderIndex,
                          model: nn.Module,
                          lossFcn: nn.Module,
                          optimizer,
                          config: dict = {}):
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
    device = config.get('device', GetDeviceMulti())
    numOfEpochs = config.get('epochs', 10)
    enableSave = config.get('saveCheckpoints', True)
    checkpoint_dir = config.get('checkpointsOutDir', './checkpoints')
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
            if not (os.path.isdir(checkpoint_dir)):
                os.mkdir(checkpoint_dir)

            exampleInput = GetSamplesFromDataset(validationDataset, 1)[0][0].reshape(
                1, -1)  # Get single input sample for model saving
            modelSaveName = os.path.join(
                checkpoint_dir, modelName + '_' + AddZerosPadding(epochID + epochStart, stringLength=4))

            SaveModel(model, modelSaveName, save_mode=AutoForgeModuleSaveMode.TRACED_DYNAMO,
                      example_input=exampleInput, target_device=device)

            if swa_model != None and swa_has_improved:
                swa_model.eval()
                SaveModel(swa_model, modelSaveName + '_SWA', save_mode=AutoForgeModuleSaveMode.TRACED_DYNAMO,
                          example_input=exampleInput, target_device=device)
                swa_model.train()

        # MODEL PREDICTION EXAMPLES
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


def EvaluateModel(dataloader: DataLoader,
                  model: nn.Module,
                  lossFcn: nn.Module,
                  device=None,
                  numOfSamples: int = 10,
                  inputSample: torch.Tensor | None = None,
                  labelsSample: torch.Tensor | None = None) -> np.ndarray:
    '''Torch model evaluation function to perform inference using either specified input samples or input dataloader'''

    if device is None:
        device = GetDeviceMulti()

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
                output_loss = lossFcn(examplePredictionList[id].to(
                    device), labelSample.to(device))

                if isinstance(output_loss, dict):
                    exampleLosses[id] = output_loss.get('lossValue').item()
                else:
                    exampleLosses[id] = output_loss.item()

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
                output_loss = lossFcn(examplePredictionList[id].to(
                    device), labelSample.to(device))

                if isinstance(output_loss, dict):
                    exampleLosses[id] = output_loss.get('lossValue').item()
                else:
                    exampleLosses[id] = output_loss.item()
        else:
            raise ValueError(
                'Either both inputSample and labelsSample must be provided or neither!')

        return examplePredictions, exampleLosses, X.to(device), Y.to(device)
