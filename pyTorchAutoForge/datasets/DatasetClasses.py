import enum
from torch.utils.data import Dataset
import numpy as np
import torch
from dataclasses import dataclass

from torch.utils.data.dataset import TensorDataset
from pathlib import Path
from pyTorchAutoForge.utils import numpy_to_torch, Align_batch_dim, torch_to_numpy
from torchvision.transforms import Compose

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from abc import abstractmethod
from abc import ABCMeta


# %% Experimental code
# DEVNOTE (PC) this is an attempt to define a configuration class that allows a user to specify dataset structure to drive the loader, in order to ease the use of diverse dataset formats

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


# %% Relatively stable code 
@dataclass
class ImagesLabelsContainer:
    """
     _summary_

    _extended_summary_
    """
    images : np.ndarray | torch.Tensor
    labels : np.ndarray | torch.Tensor
    
@dataclass
class TupledImagesLabelsContainer:
    """
     _summary_

    _extended_summary_
    """
    input_tuple : tuple[np.ndarray | torch.Tensor]
    labels : np.ndarray | torch.Tensor

    def __iter__(self, idx):
        pass 

    def __getitem__(self, idx):
        pass

    def images(self):
        """
        Return the images from the input tuple.
        """
        return self.input_tuple[0] if len(self.input_tuple) > 0 else None

    
class ImagesLabelsDatasetBase(Dataset):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def load_all_from_paths(self, 
                            images_path: str | Path, 
                            labels_path: str | Path, 
                            input_vector_data: str | Path | None = None):
        """
        Load all images and labels from the specified paths. Optionally, additional input vector data can be loaded, in which case the method returns a tuple of ((images, input_vector_data), labels).
        """
        # Implement loading logic for images and labels
        raise NotImplementedError("Loading from paths is not implemented yet.")


    def load_indexed_from_path(self):
        # TODO method to load a single (image, labels) pair from disk
        pass

class ImagesLabelsCachedDataset(TensorDataset, ImagesLabelsDatasetBase):
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
        
        if not isinstance(images_labels, ImagesLabelsContainer) and images_labels is not None:
            raise TypeError("images_labels must be of type ImagesLabelsContainer or None.")

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

        # Determine automatic input_scale_factor based on type. 
        # Default is one if cannot be inferred based on type
        self.input_scale_factor = 1.0
        if images_labels.images.max() > 1.0 and images_labels.images.dtype == torch.uint8:
            self.input_scale_factor = 255.0
        elif images_labels.images.max() > 1.0 and images_labels.images.dtype == torch.uint16:
            self.input_scale_factor = 65535.0
    
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

        # Normalize to [0,1] if max > 1 and based on dtype
        img = img.float() / self.input_scale_factor

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
    

class TupledImagesLabelsCachedDataset(ImagesLabelsDatasetBase):
    def __init__(self, tupled_images_labels: TupledImagesLabelsContainer):
        """
        Initialize the TupledImagesLabelsCachedDataset with the given images and labels.

        Args:
            images_labels (TupledImagesLabelsContainer | None): Container for images and labels.
        """
        if not isinstance(tupled_images_labels, TupledImagesLabelsContainer):
            raise TypeError("tupled_images_labels must be of type TupledImagesLabelsContainer.")

        # Initialize X and Y
        self.input_tuple = tupled_images_labels.input_tuple
        self.labels = tupled_images_labels.labels

        # Verify that the input tuple and labels have the same batch size
        if len(self.input_tuple) < 1:
            raise ValueError("Input tuple must contain at least one element.")
        
        if self.input_tuple[0].shape[0] != self.labels.shape[0]:
            raise ValueError("Batch size mismatch between input tuple and labels.")

        if len(self.input_tuple) > 1:
            # Verify all elements in the input tuple have the same batch size
            for i in range(1, len(self.input_tuple)):
                if self.input_tuple[i].shape[0] != self.labels.shape[0]:
                    raise ValueError(f"Batch size mismatch between input tuple element {i} and labels.")


    def __getitem__(self, index):
        return self.input_tuple[index], self.labels[index]

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.labels)
    
                 

class ImagesLabelsDataset():
    def __init__(self, dataset_config: DatasetLoaderConfig):
        """
        Initialize the ImagesLabelsDataset with the given configuration.

        Args:
            dataset_config (DatasetLoaderConfig): Configuration for the dataset.
        """
        self.dataset_config = dataset_config

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.dataset_config.num_samples
    
    def __getitem__(self, index):
        pass

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
