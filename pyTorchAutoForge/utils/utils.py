import torch 
import numpy as np
import random
from torch.utils.data import DataLoader

from typing import Any, Literal


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

