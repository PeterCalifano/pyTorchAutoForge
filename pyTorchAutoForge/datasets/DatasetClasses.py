import enum
from re import L
from torch.utils.data import Dataset
import numpy as np
import torch
from dataclasses import dataclass

from zipp import Path
from pyTorchAutoForge.utils import numpy_to_torch
from torchvision.transforms import Compose

# %% EXPERIMENTAL: Generic Dataset class for Supervised learning - 30-05-2024
# Base class for Supervised learning datasets
# Reference for implementation of virtual methods: https://stackoverflow.com/questions/4714136/how-to-implement-virtual-methods-in-python
from abc import abstractmethod
from abc import ABCMeta

class DatasetScope(enum.Enum):
    """
    DatasetScope class to define the scope of a dataset.
    Attributes:
        TRAINING (str): Represents the training dataset.
        TEST (str): Represents the test dataset.
        VALIDATION (str): Represents the validation dataset.
    """
    TRAINING = 'train'
    TEST = 'test'
    VALIDATION = 'validation'

    def __str__(self):
        return self.value
    def __repr__(self):
        return self.value
    def __eq__(self, other):
        if isinstance(other, DatasetScope):
            return self.value == other.value


# %% Experimental code
# DEVNOTE (PC) this is an attempt to define a configuration class that allows a user to specify dataset structure to drive the loader, in order to ease the use of diverse dataset formats

class ptaf_dtype(enum.Enum):
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    FLOAT32 = "single"
    FLOAT64 = "double"

    # DOUBT (PC) can it be convertible to torch and numpy types directly?


@dataclass
class DatasetLoaderConfig():
    """
    DatasetStructureConfig _summary_

    _extended_summary_
    """
    # Required fields
    # Generics
    dataset_name: str

    # Labels
    label_size: int
    num_samples: int = 0
    label_folder_name: str = ""

    # Optional
    # Generics
    dataset_root_path: str = "."
    hostname: str = "localhost"  # Default is local machine

    # Additional details/options
    max_num_samples: int = -1  # If -1, all samples are loaded


@dataclass
class ImagesDatasetConfig(DatasetLoaderConfig):
    image_folder: str = ""
    image_format: str = "png"
    image_dtype: type | torch.dtype = np.uint8


@dataclass
class ImagesLabelsContainer:
    """
     _summary_

    _extended_summary_
    """
    images : np.ndarray | torch.Tensor
    labels : np.ndarray | torch.Tensor
    
    
class ImagesLabelsDataset(Dataset):
    """
    ImagesLabelsDataset _summary_

    _extended_summary_

    :param Dataset: _description_
    :type Dataset: _type_
    """

    def __init__(self, images_labels: ImagesLabelsContainer | None = None, # type: ignore
                 transforms: torch.nn.Module | Compose | None = None,  
                 images_path: str | None = None, 
                 labels_path: str | None = None) -> None:
        
        # Store input and labels sources
        if images_labels is None and (images_path is None or labels_path is None):
            raise ValueError("Either images_labels container or both images_path and labels_path must be provided.")
        
        elif not (images_path is None or labels_path is None):
            # Load dataset from paths
            images_labels: ImagesLabelsContainer = self.load_from_paths(images_path, labels_path)

        if images_labels is None:
            raise ValueError("images_labels container is None after loading from paths. Something may have gone wrong. Report this issue please.")
        
        self.images = numpy_to_torch(images_labels.images)
        self.labels = numpy_to_torch(images_labels.labels)

        # Initialize transform objects
        self.transforms = transforms

    def __len__(self):
        # Number of images is batch dimension
        return np.shape(self.images)[0]


    # TODO investigate difference between __getitem__ and __getitems__
    def __getitem__(self, index):
        # Get data
        image = self.images[index, :, :, :]
        label = self.labels[index, :]

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label

    # Batch fetching
    def __getitems__(self, list_of_indices):

        # Create a numpy array from the list of indices
        indices = np.array(list_of_indices)

        # Get data
        image = self.images[indices, :, :, :] if self.images.dim() == 4 else self.images[indices, :, :]
        label = self.labels[indices, :]

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label
    
    def load_from_paths(self, images_path:str, labels_path:str) -> ImagesLabelsContainer:
        images, labels = [], [] # TODO: Implement loading logic for images and labels
        return ImagesLabelsContainer(images, labels)


# TODO function to rework as method of ImagesLabelsDataset
def LoadDataset(datasetID: int | list[int], datasetsRootFolder: str, hostname: str, limit: int = 0):
    # See Gears, reworked function is there
    pass 

# %% EXPERIMENTAL STUFF
# TODO: python Generics to implement? EXPERIMENTAL
class GenericSupervisedDataset(Dataset, metaclass=ABCMeta):
    """
    A generic dataset class for supervised learning.

    This class serves as a base class for supervised learning datasets. It 
    provides a structure for handling input data, labels, and dataset types 
    (e.g., training, testing, validation). Subclasses must implement the 
    abstract methods to define specific dataset behavior.

    Args:
        input_datapath (str): Path to the input data.
        labels_datapath (str): Path to the labels data.
        dataset_type (str): Type of the dataset (e.g., 'train', 'test', 'validation').
        transform (callable, optional): A function/transform to apply to the input data. Defaults to None.
        target_transform (callable, optional): A function/transform to apply to the target labels. Defaults to None.
    """

    def __init__(self, input_datapath: str, labels_datapath: str,
                 dataset_type: str, transform=None, target_transform=None):

        # Store input and labels sources
        self.labels_dir = labels_datapath
        self.input_dir = input_datapath

        # Initialize transform objects
        self.transform = transform
        self.target_transform = target_transform

        # Set the dataset type (train, test, validation)
        self.dataset_type = dataset_type

    def __len__(self):
        return len()  # TODO

    @abstractmethod
    def __getLabelsData__(self):
        raise NotImplementedError()
        # Get and store labels vector
        self.labels  # TODO: "Read file" of some kind goes here. Best current option: write to JSON

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()
        return inputVec, label
