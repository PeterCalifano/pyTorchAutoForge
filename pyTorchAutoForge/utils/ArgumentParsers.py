import argparse
from pyTorchAutoForge.utils import GetDeviceMulti


# Base class
class ModelOptimizationParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Config file with custom action to stop further parsing
class StopParsingAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        
        setattr(namespace, self.dest, values)
        parser.exit(message="Configuration file provided, parameters not provided to the parser will be read from it instead of default values.\n")
        
        # TODO add loading of params from config file

###################################################
# %% PTAF Optimization and Hyperparameters tuning modules parser
PTAF_training_parser = ModelOptimizationParser(
    description="CLI configuration options for pyTorchAutoForge Optimization and Hyperparameters tuning module.")

PTAF_training_parser.add_argument(
    '--config_file', type=str, default=None,
    action=StopParsingAction,
    help='Path to the configuration file; stops further argument parsing if provided')

# Parser base arguments
# Dataset processing
PTAF_training_parser.add_argument("--augment_validation_set", type=bool,
                                  default=False, help="Whether to augment the validation set.")

PTAF_training_parser.add_argument("--evaluation_dataset", type=str,
                                  default=None, help="Path to the evaluation dataset.")

## Training hyperparameters
# Batch size
PTAF_training_parser.add_argument(
    '--batch_size', type=int, default=16, help='Batch size for training')

# Number of epochs
PTAF_training_parser.add_argument(
    '--num_epochs', type=int, required=True, help='Number of epochs for training')

# Initial learning rate
PTAF_training_parser.add_argument(
    '--initial_lr', type=float, default=1E-4, help='Initial learning rate')

# Keep best strategy
PTAF_training_parser.add_argument(
    '--keep_best', type=bool, default=True, help='Keep the best model during training')

## Tracking and storage
# Checkpoint path
PTAF_training_parser.add_argument(
    '--checkpoint_path', type=str, default=None, help='Path to load a model checkpoint')

# Output folder for artifacts
PTAF_training_parser.add_argument(
    '--artifacts_folder', type=str, default='checkpoints', help='Output folder for artifacts and checkpoints savings')

# Mlflow tracking URI
PTAF_training_parser.add_argument(
    '--mlflow_tracking_uri', type=str, default=None, help='MLflow tracking URI')
# Mlflow experiment name
PTAF_training_parser.add_argument(
    '--mlflow_exper_name', type=str, default=None, help='MLflow experiment name')

# DOUBT, what are the actions? Can it accept false in that case?
PTAF_training_parser.add_argument('--device', type=str, default=GetDeviceMulti(),
                                  help='Device to use for training (e.g., "cuda" or "cpu")')

PTAF_training_parser.add_argument(
    '--save_sample_predictions', type=bool, default=True, help='Save sample predictions during training')

## Hyperparameters tuning
PTAF_training_parser.add_argument('--auto_hypertuning', action='store_true',
                                     default=False, help='Activate automatic hyperparameter tuning mode')

PTAF_training_parser.add_argument('--optuna_study_name', type=str, default="HyperparamsTuningDefaultStudy", help='Optuna study name for hyperparameter tuning')

PTAF_training_parser.add_argument('--optuna_storage', type=str, default="sqlite:///default_optuna.db", help='Optuna storage for hyperparameter tuning')
####################################################

# TODO add functionality to load and return configuration through the parser loading from file?

