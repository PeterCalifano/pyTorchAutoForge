import enum
from torch.utils.data import Dataset
import enum

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

# TODO: python Generics to implement?
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
