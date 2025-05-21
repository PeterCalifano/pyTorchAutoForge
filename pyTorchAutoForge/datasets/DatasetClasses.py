import enum
from torch.utils.data import Dataset
import numpy as np
import torch, os, sys
from dataclasses import dataclass

from torch.utils.data.dataset import TensorDataset
from pathlib import Path
from pyTorchAutoForge.utils import numpy_to_torch, Align_batch_dim, torch_to_numpy
from torchvision.transforms import Compose

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from abc import ABC, abstractmethod
from abc import ABCMeta
from typing import Any, Callable, Literal
import yaml
from PIL import Image

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
                        normalization_type: NormalizationType,
                        params: dict | None = None):
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
        data_matrix_: np.ndarray = torch_to_numpy(data_matrix).copy()
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
            raise ValueError(
                "NormalizationType.RESOLUTION requires 'resx', 'resy', and 'normalization_indices' parameters.")

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
    images: np.ndarray | torch.Tensor
    labels: np.ndarray | torch.Tensor


@dataclass
class TupledImagesLabelsContainer:
    """
     _summary_

    _extended_summary_
    """
    input_tuple: tuple[np.ndarray | torch.Tensor]
    labels: np.ndarray | torch.Tensor

    def __iter__(self, idx):
        pass

    def __getitem__(self, idx):
        pass

    def images(self):
        """
        Return the images from the input tuple.
        """
        return self.input_tuple[0] if len(self.input_tuple) > 0 else None


class ImagesLabelsDatasetBase(Dataset, ABC):
    """
    A PyTorch Dataset for loading images with associated scene and camera metadata and labels.
    Supports 'pil' or 'cv2' backends for image loading.
    Scene metadata and labels are loaded per image; camera parameters once at init.
    """

    def __init__(self,
                 image_paths: list[str],
                 camera_path: str | None = None,
                 transform: Callable[[Image.Image], Any] | None = None,
                 image_backend: Literal['pil', 'cv2'] = 'cv2'
                 ):
        """
        Args:
            image_paths: list of file paths to image metadata YAML files.
            camera_path: file path to camera parameters YAML file.
            transform: optional callable to apply to PIL Image samples.
        """
        super().__init__()

        self.image_backend = image_backend

        if image_backend == 'cv2':
            import cv2
            self.cv2 = cv2
        elif image_backend == 'pil':
            from PIL import Image
            self.pil = Image
        
        # Store paths to images
        self.image_paths = image_paths

        if camera_path is not None:
            # Load camera parameters
            self.camera_params = self._load_yaml(camera_path)
        else:
            # Initialize camera parameters as empty dict
            self.camera_params = {}

        # Store transform function
        # TODO input is a PIL image, but may not be the best format to load
        self.transform = transform

        # Cache attribute for processed labels
        self._label_cache: dict[str, Any] = {}

    def _load_yaml(self, path: str) -> dict[str, Any]:
        """
        Load and return YAML content as dict.
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def from_directory(cls,
                       root_dir: str,
                       image_meta_ext: str = '.yml',
                       camera_meta_filename: str = 'camera.yml',
                       transform: Callable[[Image.Image], Any] | None = None,
                       image_backend: Literal['pil', 'cv2'] = 'cv2'
                       ) -> "ImagesLabelsDatasetBase":
        """
        Scan root_dir for image metadata files and load global camera params.
        """
        import glob
        pattern = os.path.join(root_dir, f"*{image_meta_ext}")
        image_paths = sorted(glob.glob(pattern))
        camera_path = os.path.join(root_dir, camera_meta_filename)

        return cls(image_paths, camera_path, transform, image_backend)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_indexed_from_path(self):
        # TODO method to load a single (image, labels) pair from disk
        pass

    def load_image(self, image_path: str) -> dict[str, Any]:
        """
        Load image via selected backend, plus metadata and camera params.
        """
        scene_meta = self._load_yaml(image_path)
        
        img_filename = scene_meta.get('image')
        if img_filename is None:
            raise KeyError(f"'image' key not found in {image_path}")
        img_path = os.path.join(os.path.dirname(image_path), img_filename)

        # Image loading
        if self.image_backend == 'pil':
            if self.pil is None:
                raise ImportError("PIL is not installed")
            image = self.pil.open(img_path)
        else:
            # cv2 backend: loads BGR by default
            image = self.cv2.imread(img_path, self.cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(
                    f"Failed to load image {img_path} with cv2")

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'scene_metadata': scene_meta,
            'camera_parameters': self.camera_params
        }

    def load_labels(self, image_path: str) -> Any:
        """
        Load and process labels with caching. Override _process_labels in subclasses.
        """
        if image_path in self._label_cache:
            return self._label_cache[image_path]

        base, _ = os.path.splitext(image_path)
        raw_label_path = f"{base}_label.yml"
        if not os.path.exists(raw_label_path):
            raise FileNotFoundError(f"Label file not found: {raw_label_path}")

        with open(raw_label_path, 'r') as f:
            raw_labels = yaml.safe_load(f) or {}

        processed = self._process_labels(raw_labels)
        self._label_cache[image_path] = processed
        return processed

    def _process_labels(self, raw_labels: dict[str, Any]) -> Any:
        """
        Placeholder for label processing. Override in subclass.
        """
        return raw_labels


class ImagesLabelsCachedDataset(TensorDataset, ImagesLabelsDatasetBase):
    """
    ImagesLabelsCachedDataset _summary_

    _extended_summary_

    :param Dataset: _description_
    :type Dataset: _type_
    """

    def __init__(self, images_labels: ImagesLabelsContainer | None = None,  # type: ignore
                 transforms: torch.nn.Module | Compose | None = None,
                 images_path: str | None = None,
                 labels_path: str | None = None) -> None:

        if not isinstance(images_labels, ImagesLabelsContainer) and images_labels is not None:
            raise TypeError(
                "images_labels must be of type ImagesLabelsContainer or None.")

        # Store input and labels sources
        if images_labels is None and (images_path is None or labels_path is None):
            raise ValueError(
                "Either images_labels container or both images_path and labels_path must be provided.")

        elif not (images_path is None or labels_path is None):
            # Load dataset from paths
            raise NotImplementedError(
                "Loading from paths is not implemented yet.")
            images_labels: ImagesLabelsContainer = self.load_from_paths(
                images_path, labels_path)

        if images_labels is None:
            raise ValueError(
                "images_labels container is None after loading from paths. Something may have gone wrong. Report this issue please.")

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
                print(
                    f'\033[93mAutomatic alignment failed due to error: {err}. Please check the input dataset.\033[0m')
                raise ValueError(
                    f"Automatic alignment failed due to error {err}. Please check the input dataset.")

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
    # def __getitem__(self, index):
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
            raise TypeError(
                "tupled_images_labels must be of type TupledImagesLabelsContainer.")

        # Initialize X and Y
        self.input_tuple = tupled_images_labels.input_tuple
        self.labels = tupled_images_labels.labels

        # Verify that the input tuple and labels have the same batch size
        if len(self.input_tuple) < 1:
            raise ValueError("Input tuple must contain at least one element.")

        if self.input_tuple[0].shape[0] != self.labels.shape[0]:
            raise ValueError(
                "Batch size mismatch between input tuple and labels.")

        if len(self.input_tuple) > 1:
            # Verify all elements in the input tuple have the same batch size
            for i in range(1, len(self.input_tuple)):
                if self.input_tuple[i].shape[0] != self.labels.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch between input tuple element {i} and labels.")

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


