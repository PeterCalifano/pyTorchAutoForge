
import torch 
import numpy as np
import random
from torch.utils.data import DataLoader

import time
from functools import wraps
from typing import Any, Literal
from collections.abc import Callable

# Interfaces between numpy and torch tensors
def torch_to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:

    if isinstance(tensor, torch.Tensor):
        # Convert to torch tensor to numpy array
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    elif isinstance(tensor, np.ndarray):
        # Return the array itself
        return tensor
    else:
        raise ValueError("Input must be a torch.Tensor or np.ndarray")

def numpy_to_torch(array: torch.Tensor | np.ndarray) -> torch.Tensor:

    if isinstance(array, np.ndarray):
        # Convert numpy array to torch tensor
        return torch.from_numpy(array)

    elif isinstance(array, torch.Tensor):
        # Return the tensor itself
        return array
    else :
        raise ValueError("Input must be a torch.Tensor or np.ndarray")

# GetDevice:
def GetDevice() -> Literal['cuda:0', 'cpu', 'mps']:
    '''Function to get working device. Once used by most modules of pyTorchAutoForge, now replaced by the more advanced GetDeviceMulti(). Prefer the latter one to this method.'''
    return ('cuda:0'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available()
            else 'cpu')

# %% Function to extract specified number of samples from dataloader - 06-06-2024
# ACHTUNG: TO REWORK USING NEXT AND ITER!
def GetSamplesFromDataset(dataloader: DataLoader, numOfSamples: int = 10):

    samples = []
    for batch in dataloader:
        for sample in zip(*batch):  # Construct tuple (X,Y) from batch
            samples.append(sample)

            if len(samples) == numOfSamples:
                return samples

    return samples


# %% Other auxiliary functions - 09-06-2024
def AddZerosPadding(intNum: int, stringLength: str = '4'):
    '''Function to add zeros padding to an integer number'''
    return f"{intNum:0{stringLength}d}"  # Return strings like 00010

def getNumOfTrainParams(model):
    '''Function to get the total number of trainable parameters in a model'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def SplitIdsArray_RandPerm(array_of_ids, training_perc, validation_perc, rng_seed=0, *args):
    """
    Randomly split an array of IDs into three sets (training, validation, and testing)
    based on the input percentages. Optionally extracts values into any number of input
    arrays (using *args) generating three sets corresponding to the IDs.

    Parameters:
    - array_of_ids (torch.Tensor): Array of IDs to be split.
    - training_perc (float): Percentage of IDs for the training set.
    - validation_perc (float): Percentage of IDs for the validation set.
    - rng_seed (int): Random seed for reproducibility.
    - *args (torch.Tensor): Additional arrays to be split based on the IDs.

    Returns:
    - training_set_ids (torch.Tensor): IDs for the training set.
    - validation_set_ids (torch.Tensor): IDs for the validation set.
    - testing_set_ids (torch.Tensor): IDs for the testing set.
    - varargout (list): List of dictionaries containing split arrays for each input in *args.
    """

    # Set random seed for reproducibility
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    array_len = len(array_of_ids)

    # Shuffle the array to ensure randomness
    shuffled_ids_array = array_of_ids[torch.randperm(array_len)]

    # Calculate the number of elements for each set
    num_in_set1 = round(training_perc * array_len)
    num_in_set2 = round(validation_perc * array_len)

    # Ensure that the sum of num_in_set1 and num_in_set2 does not exceed the length of the array
    if num_in_set1 + num_in_set2 > array_len:
        raise ValueError(
            'The sum of percentages exceeds 100%. Please adjust the percentages.')

    # Assign elements to each set
    training_set_ids = shuffled_ids_array[:num_in_set1]
    validation_set_ids = shuffled_ids_array[num_in_set1:num_in_set1 + num_in_set2]
    testing_set_ids = shuffled_ids_array[num_in_set1 + num_in_set2:]

    varargout = []

    if args:
        # Optionally split input arrays in *args
        for array in args:
            if array.size(1) != array_len:
                raise ValueError(
                    'Array to split does not match length of array of IDs.')

            # Get values corresponding to the IDs
            training_set = array[:, training_set_ids]
            validation_set = array[:, validation_set_ids]
            testing_set = array[:, testing_set_ids]

            tmp_dict = {
                'trainingSet': training_set,
                'validationSet': validation_set,
                'testingSet': testing_set
            }

            varargout.append(tmp_dict)

    return training_set_ids, validation_set_ids, testing_set_ids, varargout

def timeit_averaged(num_trials: int = 10) -> Callable:
    def timeit_averaged_(fcn_to_time: Callable) -> Callable:
        """
        Function decorator to perform averaged timing of a Callable object.
        This decorator measures the execution time of the decorated function over a number of trials
        and prints the average execution time.
        :param fcn_to_time: The function to be timed.
        :param num_trials: The number of trials to average the timing over. (defualt=10)
        :return: The wrapped function with timing functionality.
        """
        @wraps(fcn_to_time)
        def wrapper(*args, **kwargs):

            # Perform timing of the function using best counter available in time module
            total_elapsed_time = 0.0

            print(f'Timing function "{fcn_to_time.__name__}" averaging {num_trials} trials...')

            for idT in range(num_trials):
                start_time = time.perf_counter()
                result = fcn_to_time(*args, **kwargs) # Returns Any
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"\rFunction call {idT} took {elapsed_time:.6f} seconds")

                # Calculate the elapsed time
                total_elapsed_time += elapsed_time
            
            # Calculate the average elapsed time
            average_elapsed_time = total_elapsed_time / num_trials
            print(f"\nAverage time over {num_trials} trials: {average_elapsed_time:.6f} seconds")

            return result
        return wrapper
    return timeit_averaged_

def timeit_averaged_(fcn_to_time: Callable, num_trials: int = 10, *args, **kwargs) -> float:
    # Perform timing of the function using best counter available in time module
    total_elapsed_time = 0.0

    for idT in range(num_trials):

        start_time = time.perf_counter()
        out = fcn_to_time(*args, **kwargs)  # Returns Any
        end_time = time.perf_counter()

        total_elapsed_time += end_time - start_time

    # Calculate the average elapsed time
    average_elapsed_time = total_elapsed_time / num_trials    
    return average_elapsed_time

@timeit_averaged(2)
def dummy_function(): 
    print("Dummy function called")
    time.sleep(1)

def test_timeit_averaged():
    print("Testing timeit_averaged wrapper...")
    # Example usage
    dummy_function()

def test_SplitIdsArray_RandPerm():
    # Example usage
    N = 100
    array_of_ids = torch.arange(0, N + 1, dtype=torch.int32)
    training_perc = 0.2  # 20%
    validation_perc = 0.3  # 30%
    rng_seed = 42

    # Example additional arrays
    additional_array1 = torch.rand((5, len(array_of_ids)))
    additional_array2 = torch.rand((3, len(array_of_ids)))

    training_set_ids, validation_set_ids, testing_set_ids, varargout = SplitIdsArray_RandPerm(
        array_of_ids, training_perc, validation_perc, rng_seed, additional_array1, additional_array2
    )

    print('Training Set IDs:', training_set_ids)
    print('Validation Set IDs:', validation_set_ids)
    print('Testing Set IDs:', testing_set_ids)
    print('Varargout:', varargout)

if __name__ == '__main__':
    test_SplitIdsArray_RandPerm()
    test_timeit_averaged()

