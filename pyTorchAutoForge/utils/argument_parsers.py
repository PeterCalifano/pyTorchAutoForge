import argparse as argp
from pyTorchAutoForge.utils import GetDeviceMulti
from typing import Any
import yaml

# Auxiliary functions 
def _load_yaml(Path: str) -> dict[str, Any]:
    """Load YAML file and return as dict."""
    with open(Path, 'r') as f:
        return yaml.safe_load(f) or {}

def _merge_config(Args: argp.Namespace, Config: dict[str, Any]) -> None:
    """Merge YAML config dict into argp Namespace, overriding existing attrs."""
    for key, value in Config.items():
        if hasattr(Args, key):
            setattr(Args, key, value)

# %% Base classes
class ConfigArgumentParser(argp.ArgumentParser):
    """
    ArgumentParser that supports loading defaults from a YAML config file.
    Providing -y/--yaml_config will override all other args.
    """

    def __init__(
        self,
        *args: Any,
        yaml_arg_name: str = 'yaml_config',
        **kwargs: Any
    ) -> None:
        
        super().__init__(*args, **kwargs)
        self._yaml_arg_name: str = yaml_arg_name
        self.add_argument(
            '-y', f'--{self._yaml_arg_name}',
            dest=self._yaml_arg_name,
            type=str,
            help='Path to YAML config file to override arguments'
        )

    def ParseArgs(self, args: Any | None = None, namespace: argp.Namespace | None = None) -> argp.Namespace:

        # First parse known to check for YAML
        preliminary, _ = super().parse_known_args(args, namespace)
        yaml_path: str | None = getattr(
            preliminary, self._yaml_arg_name, None)
        
        if yaml_path:
            config: dict[str, Any] = _load_yaml(yaml_path)

            # Build fresh namespace with defaults
            new_ns: argp.Namespace = argp.Namespace()

            for action in self._actions:
                if action.dest != 'help':
                    setattr(new_ns, action.dest, action.default)
                    
            _merge_config(new_ns, config)

            return new_ns
        
        # Fallback to standard parse_args
        return super().parse_args(args, namespace)

    # Allow both parse_args and parse_args aliasing
    def parse_args(self, args: Any | None = None, namespace: argp.Namespace | None = None) -> argp.Namespace:
        return self.ParseArgs(args, namespace)


# Specific base parsers
class ModelOptimizationParser(argp.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Config file with custom action to stop further parsing
class StopParsingAction(argp.Action):
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
PTAF_training_parser.add_argument("--augment_validation_set", 
                                  action='store_true', 
                                  help="Enable augmentation applied to the validation set.")


def parse_eval_dataset(arg):
    
    # If there's a comma, split by comma
    if isinstance(arg, tuple):
        return arg
    elif ',' in arg:
        return tuple(arg.split(','))
    # If there's a space, split by space
    elif ' ' in arg:
        return tuple(arg.split())
    # Otherwise, return as a single-element tuple
    else:
        return (arg,)


PTAF_training_parser.add_argument("--evaluation_dataset", type=parse_eval_dataset,
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
    '--mlflow_experiment_name', type=str, default=None, help='MLflow experiment name')

PTAF_training_parser.add_argument('--device', type=str, default=None,
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

####################################################
# %% Auxiliary utilities
def ParseShapeString(shape_str: str, delimiter: str = ',') -> tuple[int, ...]:
    """Convert a comma-separated string into a tuple of ints."""
    parts = shape_str.split(delimiter)
    try:
        return tuple(int(p.strip()) for p in parts)
    except ValueError:
        raise argp.ArgumentTypeError(
            f"Invalid shape '{shape_str}'. Expected comma-separated integers, e.g. '1,3,224,224'"
        )
