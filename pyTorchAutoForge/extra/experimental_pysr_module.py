from pysr import PySRRegressor
import numpy as np
import h5py
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import sympy, os
from pandas import DataFrame
import copy
from dataclasses import dataclass, field
from pyTorchAutoForge.utils import torch_to_numpy
from pyTorchAutoForge.setup import BaseConfigClass
from typing import Any
import pprint

# TODO move these classes to dedicated PTAF module
@dataclass
class PySymbolicRegrConfig(BaseConfigClass):
    # Model checkpoint
    checkpoint_path : str | None = None
    update_save_settings: bool = False
    warm_start : bool = True
    # Evolutationary search params
    maxsize: int = 35
    niterations: int = 20000
    population_size: int = 75
    populations: int = 3*28
    annealing: bool = True
    parsimony : float = 0.0
    optimize_probability: float = 0.15
    warmup_maxsize_by: int | float = 0
    model_selection :str = "best"
    elementwise_loss: str = "loss(prediction, target) = (prediction - target)^2"
    # Data processing
    batching: bool = True
    batch_size: int = 100
    select_k_features: int | None = None
    # Operators
    binary_operators: list[str] = field(default_factory=lambda: ["+", "*", "/", "-", "^"])
    unary_operators: list[str] = field(default_factory=lambda: ["cos", "exp", "sqrt", "log", "abs"])
    # Constraints
    constraints: dict[str, tuple[int, int] | list[int] | int] = field(default_factory=lambda: {
        "^": (-1, 20),
        "log": 20,
        "sqrt": 20,
        "cos": 20,
        "exp": 20,
        "abs": 30
    })
    nested_constraints: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "cos": {"cos": 0, "exp": 1},
        "exp": {"cos": 1, "exp": 0},
        "abs": {"abs": 0},
        "sqrt": {"sqrt": 0, "log": 1, "cos": 1},
        "log": {"log": 1, "exp": 0},
    })
    # Complexity
    complexity_of_operators: dict[str, int] = field(default_factory=lambda: {"cos": 1})
    # Implementation options
    turbo: bool = False

# Predefined configurations 
@dataclass
class PySymbolicRegrPolyOnlyConfig(PySymbolicRegrConfig):
    maxsize: int = 50
    niterations: int = 20000
    population_size: int = 100
    populations: int = 35  # 90
    binary_operators: list[str] = field(default_factory=lambda: ["+", "*", "/", "-", "^"])
    unary_operators: list[str] = field(default_factory=lambda: ["sign", "abs"])
    constraints: dict[str, tuple[int, int] | list[int] | int] = field(default_factory=lambda: {"^": (-1, 30)})
    nested_constraints: dict[str, dict[str, int]] = field(default_factory=dict)
    complexity_of_operators: dict[str, int] = field(default_factory=lambda: {"^": 0, "+": 0, "-": 0, "*": 0, "/": 1})
    parsimony: float = 0
    model_selection: str = "accuracy"
    optimize_probability: float = 0.2
    optimizer_iterations: int = 20

class OptimizationHelperPySR():
    """
    OptimizationHelperPySR is a helper class designed to facilitate the use of the PySRRegressor model.

    This class provides methods for initializing, optimizing, and updating the settings of a PySRRegressor model.
    It also handles data preprocessing, such as converting PyTorch tensors to NumPy arrays, and supports loading
    model checkpoints for warm-starting the optimization process.

    Attributes:
        config (PySymbolicRegrConfig): Configuration object containing settings for the PySRRegressor model.
        model (PySRRegressor | None): Instance of the PySRRegressor model, if provided or loaded from a checkpoint.
        input_data (np.ndarray | torch.Tensor | None): Input data for the model, converted to a NumPy array if necessary.
        output_data (np.ndarray | torch.Tensor | None): Output data for the model, converted to a NumPy array if necessary.
    """
    def __init__(self, config: PySymbolicRegrConfig, model: PySRRegressor | None = None, input_data: np.ndarray | torch.Tensor | None = None, output_data: np.ndarray | torch.Tensor | None = None) -> None:
        # Initialize attributes values
        self.config = config
        self.model = model

        # Convert torch tensors to numpy arrays if necessary
        self.input_data = torch_to_numpy(input_data) if input_data is not None else None
        self.output_data = torch_to_numpy(output_data) if output_data is not None else None

        # If model is provided, override any checkpoint indicated in the config
        if model is not None:
            return

        # Reload pySR model if checkpoint path is provided
        if self.config.checkpoint_path is not None:
            model = PySRRegressor.from_file(run_directory=self.config.checkpoint_path)
            print('Model state loaded from pkl: ', model)

            if self.config.update_save_settings:
                self.update_model_settings(config=config, model=model)
            else:
                print('Optimization will use settings defined in checkpoint file.')
        else: 
            # Build pySR model from scratch
            print('New PySR model created from scratch with provided configuration:')
            pprint.pprint(self.config.__dict__)
            self.model = PySRRegressor()
            self.update_model_settings(config=config, model=self.model)

    def optimize(self, input_data: np.ndarray | torch.Tensor | None = None, output_data: np.ndarray | torch.Tensor | None = None):
        """
        optimize_model performs optimization using the PySRRegressor model.

        This method takes input and output data, updates the internal data if provided,
        and starts the evolutionary fitting process using the PySRRegressor model.

        :param input_data: Input data for the model, defaults to None
        :type input_data: np.ndarray | torch.Tensor | None, optional
        :param output_data: Output data for the model, defaults to None
        :type output_data: np.ndarray | torch.Tensor | None, optional
        """
        assert isinstance(self.model, PySRRegressor), "Model is not a PySRRegressor instance. Please provide a valid model."

        if input_data is not None:
            if self.input_data is not None:
                print("\033[93mWarning: Overwriting existing input data.\033[0m")
            self.input_data = torch_to_numpy(input_data)
                    
        if output_data is not None:
            if self.output_data is not None:
                print("\033[93mWarning: Overwriting existing output data.\033[0m")
            self.output_data = torch_to_numpy(output_data)

        # Start fitting
        print('Starting evolutionary and fitting process! :D')
        self.model.fit(self.input_data, self.output_data)
        print('Fitting process completed!')

    def update_model_settings(self, config: PySymbolicRegrConfig | None = None, model : PySRRegressor | None = None):
        """
        Updates the settings of the PySRRegressor model using the provided configuration.

        This method allows updating the model's attributes based on the configuration
        provided either during the instantiation of the helper class or passed explicitly
        to this method.

        :param config: Configuration object containing the settings to update the model with.
                   If not provided, the configuration used during class instantiation will be used.
        :type config: PySymbolicRegrConfig | None, optional
        :param model: PySRRegressor model instance to update. If not provided, the model
                  used during class instantiation will be updated.
        :type model: PySRRegressor | None, optional
        :raises ValueError: If neither a model nor a configuration is provided.
        """
        if model is None and self.model is None:
            raise ValueError("No model provided for update as input to this method or at helper class instantiation.")
    
        if config is None and self.config is None:
            raise ValueError("No configuration provided for model update as input to this method or at helper class instantiation.")
        
        # Use provided configuration if available
        if config is None and self.config is not None:
            config = self.config

        if model is None and self.model is not None:
            model = self.model
            
        # Update model settings with specified configuration
        print('Updating PySRRegressor with specified settings...')
        params_to_copy = ["extra_sympy_mappings", "binary_operators", "unary_operators", 
            "batch_size", "niterations", "complexity_of_operators", 
            "constraints", "nested_constraints", "elementwise_loss", 
            "parsimony", "model_selection", "optimize_probability", 
            "optimizer_iterations", "population_size", 
            "populations", "annealing", "warmup_maxsize_by", "turbo"]

        # Update model attributes based on config
        for param in params_to_copy:
            if hasattr(config, param):
                setattr(self.model, param, getattr(config, param))

#%% LEGACY CODES BELOW
def RunRegression(input_data: np.ndarray, output_data: np.ndarray, checkpoint_path: str | None = None, update_save_settings: bool = False):
    # Setup pySR regression model
    model = PySRRegressor(
        maxsize=35,
        niterations=20000,
        population_size=75,
        populations=3*28,  # 3*number of cores
        batching=True,
        batch_size=100,
        binary_operators=["+", "*", "/", "-", "^"],
        unary_operators=[
            "cos",
            "exp",
            "sqrt",
            "log",
            "abs",
            # ^ Custom operator (julia syntax)
        ],
        denoise=False,
        constraints={"^": (-1, 20), "log": (20), "sqrt": (20),
                        "cos": (20), "exp": (20), "abs": (30)},
        # Avoid nesting of trigonometric functions
        nested_constraints={"cos": {"cos": 0, "exp": 1}, "exp": {
            "cos": 1, "exp": 0}, "abs": {"abs": 0}, "sqrt": {"sqrt": 0, "log": 1, "cos": 1}, "log": {"log": 1, "exp": 0}, "exp": {"log": 0}},
        complexity_of_operators={"exp": 1,
                                    "sqrt": 1,
                                    "log": 1,
                                    "^": 0,       
                                    "+": 0,
                                    "-": 0,
                                    "*": 0,
                                    "/": 1,
                                    "cos": 2,
                                    },
        #select_k_features=4,
        turbo=False,
        elementwise_loss="loss(prediction, target) = 0.85*(prediction - target)^2 + 0.15*abs((prediction - target))",
        warm_start=True,
        parsimony=0.000001,
        model_selection="accuracy",
        optimize_probability=0.15,
        verbosity=1,
        annealing=True,
        warmup_maxsize_by=0,
    )

    if checkpoint_path is not None:
        # Copy model from scratch
        model_empty = copy.copy(model)

        # Load model from checkpoint
        model = PySRRegressor.from_file(run_directory=checkpoint_path)
        print('Model state loaded from pkl: ', model)

        if update_save_settings:
            print('Updating state of loaded checkpoint with new settings...')
            params_to_copy = ["extra_sympy_mappings", "binary_operators", "unary_operators", "batch_size", "niterations", "complexity_of_operators", "constraints", "nested_constraints", "elementwise_loss", "parsimony", "model_selection", "optimize_probability", "optimizer_iterations"]

            # Copy attributes from model_empty to model
            for param in params_to_copy:
                setattr(model, param, getattr(model_empty, param))
    else:
        print('Starting with model state from scratch: ', model)


    # Start search
    print('Starting evolutionary and fitting process! :D')
    model.fit(input_data, output_data)
    print('Fitting process completed!')

    return model

def RunRegressionPolyOnly(input_data: np.ndarray, output_data: np.ndarray, checkpoint_path: str | None = None):
    if checkpoint_path is None:
        # Setup pySR regression model
        model = PySRRegressor(
            maxsize=50,
            niterations=20000,
            population_size=300,
            populations=3*30,  # 3*number of cores
            batching=True,
            binary_operators=["+", "*", "/", "-", "^", "prod2(x,y) = x*y"],
            unary_operators=[
                "sign",
                "abs",
                # "tan",
                # "atan"
                # ^ Custom operator (julia syntax)
            ],
            constraints={"^": (-1, 1)},
            # Avoid nesting of trigonometric functions
            nested_constraints={},
            complexity_of_operators={"^": 0,       # Count exponent as 2 if you want to discourage large powers
                                     "+": 0,
                                     "-": 0,
                                     "*": 0,
                                     "/": 1,
                                     },
            # select_k_features=3,
            turbo=False,
            elementwise_loss="loss(prediction, target) = 0.75 * (prediction - target)^2 + 0.25 * abs(prediction - target)",
            warm_start=True,
            parsimony=1,
            model_selection="accuracy",
            optimize_probability=0.5,
            optimizer_iterations=20,
            extra_sympy_mappings={"prod2": lambda x, y: x*y, "prod3": lambda x, y, z: x*y*z, "prod4": lambda x, y, z, w: x*y*z*w},
            )
        print('Model state from scratch: ', model)

    else:
        # Load model from checkpoint
        model = PySRRegressor.from_file(run_directory=checkpoint_path)
        print('Model state loaded from pkl: ', model)

    # Start search
    print('Starting evolutionary and fitting process! :D')
    model.fit(input_data, output_data)
    print('Fitting process completed!')
    return model

def RunRegressionPCApreProc(input_data: np.ndarray, output_data: np.ndarray, checkpoint_path: str | None = None):

    if checkpoint_path is None:
        # Setup pySR regression model
        model = PySRRegressor(
            maxsize=25,
            niterations=20000,
            population_size=300,
            populations=3*30,  # 3*number of cores
            batching=True,
            binary_operators=["+", "*", "/", "-", "^"],
            unary_operators=[
                "cos",
                "sin",
                "exp",
                "sqrt",
                "log",
                "abs",
                #"tan",
                #"atan"
                # ^ Custom operator (julia syntax)
            ],
            constraints={"^": (-1, 1)},
            # Avoid nesting of trigonometric functions
            nested_constraints={"cos": {"cos": 0, "exp": 1, "sin":0}, "exp": {"cos": 1, "exp": 1, "sin":1}, "sin": {"cos": 0, "exp": 1, "sin":0}, "abs": {"abs": 0}},
                complexity_of_operators={"exp": 2,
                "sqrt": 2,
                "log": 2,
                "^": 0,       # Count exponent as 2 if you want to discourage large powers
                "+": 0,
                "-": 0,
                "*": 0,
                "/": 1,
                "sin": 1,
                "cos": 1,
                },
            #select_k_features=3,
            turbo=False,
            elementwise_loss="loss(prediction, target) = 0.75 * (prediction - target)^2 + 0.25 * abs(prediction - target)",
            warm_start=True,
            parsimony=1,
            model_selection = "accuracy",
            optimize_probability = 0.5,
            optimizer_iterations = 20
        )
        print('Model state from scratch: ', model)

    else:
        # Load model from checkpoint
        model = PySRRegressor.from_file(run_directory=checkpoint_path)
        print('Model state loaded from pkl: ', model)


    # Start search
    print('Starting evolutionary and fitting process! :D')
    model.fit(input_data, output_data)
    print('Fitting process completed!')
    return model
      
def TestPrediction(model: PySRRegressor | torch.nn.Module, input_vector: torch.Tensor | np.ndarray, reference_output:  torch.Tensor | np.ndarray):

    # pySR model case
    if isinstance(model, PySRRegressor):
        assert isinstance(
            input_vector, np.ndarray), "Input vector must be a numpy.ndarray if model is a PySR model."
        try:
            predictions = model.predict(input_vector)
            error_data = np.abs(predictions - reference_output)
            return error_data
        except Exception as e:
            print("Error in prediction: ", e)
            return None
    elif isinstance(model, torch.nn.Module):

        assert isinstance(
            input_vector, torch.Tensor), "Input vector must be a torch.Tensor if model is a PyTorch model."
        try:
            predictions = model(input_vector)
            error_data = np.abs(predictions - reference_output)
            return error_data
        except Exception as e:
            print("Error in prediction: ", e)


# %% Class experimental implementations
class sklearnModelManager():
    def __init__(self):
        pass 

    def load_model_checkpoint(self):
        pass

    def optimize_model(self):
        pass


