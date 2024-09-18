from torch.utils.data import Dataset


# %% EXPERIMENTAL: Generic Dataset class for Supervised learning - 30-05-2024
# Base class for Supervised learning datasets
# Reference for implementation of virtual methods: https://stackoverflow.com/questions/4714136/how-to-implement-virtual-methods-in-python
from abc import abstractmethod
from abc import ABCMeta

class GenericSupervisedDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, inputDataPath: str = 'inputData/', labelsDataPath: str = 'labelsData/',
                 datasetType: str = 'train', transform=None, target_transform=None):
        # Store input and labels sources
        self.labelsDir = labelsDataPath
        self.inputDir = inputDataPath

        # Initialize transform objects
        self.transform = transform
        self.target_transform = target_transform

        # Set the dataset type (train, test, validation)
        self.datasetType = datasetType

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
