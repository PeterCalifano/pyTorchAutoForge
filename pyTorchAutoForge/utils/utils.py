import torch 
import numpy as np
import random
from torch.utils.data import DataLoader

from typing import Any, Literal

# ComputeModelParamsStorageSize
def ComputeModelParamsStorageSize(model: torch.nn.Module) -> float:
    '''Function to compute the size of the model parameters in MB.'''

    # Sum size of all parameters in bytes
    total_bytes = sum(p.numel() * p.element_size()
                    for p in model.parameters())
    
    # Convert to MB
    size_mb = total_bytes / 1024**2
    return size_mb

# Function aligning batch size of two tensors


def Align_batch_dim(images: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
    """
        If labels has one axis whose length == images.size(0),
        assume that's the batch-axis, permute it to dim 0, and return.
        Otherwise, leave labels untouched (or raise).
    """

    batch_size = images.size(0)
    lab_shape: tuple = labels.shape

    # find all axes in labels matching batch_size
    match_axes = [i for i, L in enumerate(lab_shape) if L == batch_size]

    if not match_axes:
        raise ValueError(
            f"labels array shape {lab_shape} does not contain any axes matching batch size {batch_size}.")

    if len(match_axes) > 1:
        print(f"\033[93mWarning: found multiple axes {match_axes} "
                f"in labels matching batch‐size {batch_size}, picking first.\033[0m")

    # Pick first matching axis
    axes_id = match_axes[0]

    if axes_id == 0:
        # Already in front
        return labels

    # Build a permutation that brings ax → 0, and shifts others right
    perm = [axes_id] + [i for i in range(len(lab_shape)) if i != ax]
    print(f"\033[93mInfo: permuting labels dims {lab_shape} → "
            f"{[lab_shape[i] for i in perm]} (perm={perm})\033[0m")
    return labels.permute(*perm)


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

# TODO, move to MachineLearningGears
def ComputeRangeFromApparentRadius(apparentRadiusInPix: float | torch.Tensor, focal_length: float, range_metric_scale: float, IFOV: float) -> float | torch.Tensor:

    # Check input types validity
    assert isinstance(focal_length, float), "Focal length should be a float"
    assert isinstance(range_metric_scale,
                      float), "range_metric_scale should be a float"
    assert isinstance(IFOV, float), "IFOV should be a float"

    assert IFOV > 0, "IFOV should be positive"
    assert range_metric_scale > 0, "range_metric_scale should be positive"
    assert focal_length > 0, "focal_length should be positive"

    apparent_angular_size = apparentRadiusInPix * IFOV

    if isinstance(apparent_angular_size, float):
        range_from_pix = range_metric_scale * np.cos(apparent_angular_size) * (
            np.tan(apparent_angular_size) + focal_length/apparentRadiusInPix)

    elif isinstance(apparent_angular_size, torch.Tensor):
        range_from_pix = range_metric_scale * torch.cos(apparent_angular_size) * (
            torch.tan(apparent_angular_size) + focal_length/apparentRadiusInPix)

    return range_from_pix

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

