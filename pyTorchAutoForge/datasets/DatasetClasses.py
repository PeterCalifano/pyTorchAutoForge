import enum
from re import L
from torch.utils.data import Dataset
import numpy as np
import torch
from dataclasses import dataclass

from torch.utils.data.dataset import TensorDataset
from zipp import Path
from pyTorchAutoForge.utils import numpy_to_torch, Align_batch_dim, torch_to_numpy
from torchvision.transforms import Compose

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from abc import abstractmethod
from abc import ABCMeta

class NormalizationType(enum.Enum):
    NONE = "None"
    ZSCORE = "ZScore"
    RESOLUTION = "Resolution"
    MINMAX = "MinMax"

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
    UINT8 = "uint8"
    INT16 = "int16"
    UINT16 = "uint16"
    INT32 = "int32"
    UINT32 = "uint32"
    FLOAT32 = "single"
    FLOAT64 = "double"

    # DOUBT (PC) can it be convertible to torch and numpy types directly?

# TODO temporary version, make a unified class to handle normalizations
def NormalizeDataMatrix(data_matrix: np.ndarray | torch.Tensor, 
                        normalization_type : NormalizationType, 
                        params : dict | None = None) :
    """
    Normalize the data matrix based on the specified normalization type.

    Args:
        data_matrix (numpy.ndarray | torch.Tensor): The data matrix to be normalized.
        normalization_type (NormalizationType): The type of normalization to apply.
        params (dict | None): Additional arguments for normalization.

    Returns:
        numpy.ndarray | torch.Tensor: The normalized data matrix.
    """

    was_tensor = False

    if isinstance(data_matrix, torch.Tensor):
        was_tensor = True
        data_matrix_ : np.ndarray= torch_to_numpy(data_matrix).copy()
    elif isinstance(data_matrix, np.ndarray):
        data_matrix_ = data_matrix.copy()
    else:
        raise TypeError("data_matrix must be a numpy array or a torch tensor.")

    if normalization_type == NormalizationType.ZSCORE:

        scaler = StandardScaler(with_mean=True, with_std=True)
        data_matrix_ = scaler.fit_transform(data_matrix_)

        if was_tensor:
            data_matrix_ = numpy_to_torch(data_matrix_)

        return data_matrix_, scaler

    elif normalization_type == NormalizationType.MINMAX:

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_matrix_ = scaler.fit_transform(data_matrix_)

        if was_tensor:
            data_matrix_ = numpy_to_torch(data_matrix_)

        return data_matrix_, scaler

    elif normalization_type == NormalizationType.RESOLUTION:

        if params is None or 'resx' not in params or 'resy' not in params or 'normalization_indices' not in params:
            raise ValueError("NormalizationType.RESOLUTION requires 'resx', 'resy', and 'normalization_indices' parameters.")

        data_matrix_ = data_matrix_[:, params['normalization_indices']
                                  ] / np.array([params['resx'], params['resy']])
        
        if was_tensor:
            data_matrix_ = torch_to_numpy(tensor=data_matrix_)
            
        return data_matrix_, None
        
    elif normalization_type == NormalizationType.NONE:

        if was_tensor:
            data_matrix_ = torch_to_numpy(data_matrix_)
        
        return data_matrix_, None


        

def DeNormalizeDataMatrix(data_matrix: np.ndarray, scaler: StandardScaler | MinMaxScaler):
    pass # TODO
    return 0


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
    
    
class ImagesLabelsCachedDataset(TensorDataset):
    """
    ImagesLabelsCachedDataset _summary_

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
            raise NotImplementedError("Loading from paths is not implemented yet.")
            images_labels: ImagesLabelsContainer = self.load_from_paths(images_path, labels_path)

        if images_labels is None:
            raise ValueError("images_labels container is None after loading from paths. Something may have gone wrong. Report this issue please.")
        
        # Initialize X and Y
        images_labels.images = numpy_to_torch(images_labels.images)
        images_labels.labels = numpy_to_torch(images_labels.labels)

        # Normalize to [0,1] if max > 1 and based on dtype
        if images_labels.images.max() > 1.0 and images_labels.images.dtype == torch.uint8:
            images_labels.images = images_labels.images.float() / 255.0
            
        elif images_labels.images.max() > 1.0 and images_labels.images.dtype == torch.uint16:
            images_labels.images = images_labels.images.float() / 65535.0

        # Unsqueeze images to 4D [B, C, H, W] if 3D [B, H, W]
        if images_labels.images.dim() == 3:
            images_labels.images = images_labels.images.unsqueeze(1)

        # Check batch size (must be identical)
        if images_labels.images.shape[0] != images_labels.labels.shape[0]:
            print('\033[93mWarning: found mismatch of batch size, automatic resolution attempt using the largest dimension between images and labels...\033[0m')

            try:
                images_labels.labels = Align_batch_dim(
                    images_labels.images, images_labels.labels)

            except Exception as err:
                print(f'\033[93mAutomatic alignment failed due to error: {err}. Please check the input dataset.\033[0m')
                raise ValueError(f"Automatic alignment failed due to error {err}. Please check the input dataset.") 
            
        # Initialize base class TensorDataset(X, Y)
        super().__init__(images_labels.images, images_labels.labels)

        # Initialize transform objects
        self.transforms = transforms

        def __getitem__(self, idx):
            # Apply transform to the image and label
            img, lbl = super().__getitem__(idx)

            if self.transforms is not None:
                return self.transforms(img), self.transforms(lbl)
            
            return img, lbl
    
    # Batch fetching
    #def __getitem__(self, index):
    #    # Get data
    #    image = self.images[index, :, :, :] if self.images.dim() == 4 else self.images[index, :, :]
    #    label = self.labels[index, :]
    #    if self.transforms is not None:
    #        image, label = self.transforms(image, label)
    #    return image, label
    

    #def load_from_paths(self, images_path:str, labels_path:str) -> ImagesLabelsContainer:
    # DEVNOTE this should be implemented in a base class since in common!
    #    images, labels = [], [] # TODO: Implement loading logic for images and labels
    #    return ImagesLabelsContainer(images, labels)


# TODO function to rework as method of ImagesLabelsDataset
def LoadDataset(datasetID: int | list[int], datasetsRootFolder: str, hostname: str, limit: int = 0):
    # See Gears, reworked function is there
    pass 

# %% EXPERIMENTAL STUFF
# TODO: python Generics to implement? 
# Generic Dataset class for Supervised learning - 30-05-2024
# Base class for Supervised learning datasets
# Reference for implementation of virtual methods: https://stackoverflow.com/questions/4714136/how-to-implement-virtual-methods-in-python
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
