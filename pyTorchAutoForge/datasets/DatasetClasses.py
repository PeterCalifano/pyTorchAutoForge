import enum
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import sys
from dataclasses import dataclass, field

from torch.utils.data.dataset import TensorDataset
from pathlib import Path
from pyTorchAutoForge.utils import numpy_to_torch, Align_batch_dim, torch_to_numpy
from torchvision.transforms import Compose

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from abc import ABC, abstractmethod
from abc import ABCMeta
from typing import Any, Literal
from collections.abc import Callable
import yaml
from PIL import Image
from functools import partial
import json
import yaml
from pyTorchAutoForge.datasets.LabelsClasses import PTAF_Datakey, LabelsContainer

# DEVNOTE (PC) this is an attempt to define a configuration class that allows a user to specify dataset structure to drive the loader, in order to ease the use of diverse dataset formats

# %% Types and aliases
class ImagesDatasetType(enum.Enum):
    """
    Enumeration class for dataset types.
    """
    SEQUENCED = "ScatteredSequences"
    POINT_CLOUD = "point_cloud"  # TODO modify
    TRAJECTORY = "trajectory"  # TODO modify


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


# %% Configuration classes
@dataclass
class DatasetLoaderConfig():
    """
    DatasetStructureConfig _summary_

    _extended_summary_
    """
    # Required fields
    # Generics
    dataset_names_list: Path | str | list[str | Path]
    datasets_root_folder: str | tuple[str, ...]
    lbl_vector_data_keys: tuple[PTAF_Datakey | str, ...]

    # Labels
    # label_size: int = 0
    # num_samples: int = 0
    # label_folder_name: str = ""

    # Optional (defaults)
    hostname: str = os.uname()[1]  # Default is local machine
    labels_folder_name: str = "labels"
    lbl_dtype: type | torch.dtype = torch.float32

    # Additional details/options
    samples_limit_per_dataset: int | tuple[int, ...] = -1

    def __post_init__(self):
        """
        Post-initialization checks for the DatasetLoaderConfig.
        """
        # Convert all strings to tuple if not already
        if isinstance(self.datasets_root_folder, str):
            self.datasets_root_folder = (self.datasets_root_folder,)
        if isinstance(self.dataset_names_list, str):
            self.dataset_names_list = (self.dataset_names_list,)

        # Check all roots exist
        for root in self.datasets_root_folder:
            if not os.path.exists(root):
                raise FileNotFoundError(
                    f"Dataset root folder '{root}' does not exist.")

        # Check all datakeys are valid PTAF_Datakey
        for key in self.lbl_vector_data_keys:
            if isinstance(key, str):
                # Convert string to PTAF_Datakey if possible
                try:
                    key = PTAF_Datakey[key.upper()]
                except KeyError:
                    raise ValueError(
                        f"Invalid label data key string: {key}. Must be a valid PTAF_Datakey or a recognized string.")
            if not isinstance(key, (PTAF_Datakey)):
                raise TypeError(
                    f"lbl_vector_data_keys must be of type PTAF_Datakey or str, got {type(key)}")


@dataclass
class ImagesDatasetConfig(DatasetLoaderConfig):
    # Default options
    binary_masks_folder_name: str = "binary_masks"
    images_folder_name: str = "images"
    camera_filepath: str | None = None

    image_format: str = "png"
    image_dtype: type | torch.dtype = np.uint8
    image_backend: Literal['pil', 'cv2'] = 'cv2'

    # Additional options
    intensity_scaling_mode: Literal['none', 'dtype', 'custom'] = 'dtype'
    intensity_scale_value: float | None = None  # Used only if intensity_scaling is 'custom'

    def __post_init__(self):
        super().__post_init__()

        if intensity_scaling_mode not in ['none', 'dtype', 'custom']:
            raise ValueError(
                f"Unsupported intensity scaling mode: {self.intensity_scaling_mode}. Supported modes are 'none', 'dtype', and 'custom'.")
       
       if self.intensity_scale_value is not None and self.intensity_scaling_mode != 'custom':
           raise ValueError(
               f"intensity_scale_value must be None unless intensity_scaling_mode is 'custom'.")

######################## DEVNOTE Relatively stable code BELOW ########################
@dataclass
class DatasetPathsContainer():
    """
    DatasetPathsContainer _summary_

    _extended_summary_

    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises IndexError: _description_
    :return: _description_
    :rtype: _type_
    """
    img_filepaths: list[str]
    lbl_filepaths: list[str]

    total_num_entries: int | None = field(default=None, init=True)
    num_of_entries_in_set: list[int] | int | None = field(
        default=None, init=True)

    def __post_init__(self):
        if self.img_filepaths is None or self.lbl_filepaths is None:
            raise ValueError("Image and label file paths cannot be None.")

        if len(self.img_filepaths) != len(self.lbl_filepaths):
            raise ValueError(
                "Number of image and label file paths must match.")

        self.total_num_entries = len(self.img_filepaths)

    def __len__(self):
        """Return the total number of entries in the dataset."""
        return self.total_num_entries

    def __getitem__(self, index: int | list[int]) -> list[tuple[str, str]]:
        """Get the image and label file paths for a given index."""

        if isinstance(index, list):
            return [(self.img_filepaths[i], self.lbl_filepaths[i]) for i in index]

        if self.total_num_entries is not None:
            if index < 0 or index >= self.total_num_entries:
                raise IndexError("Index out of range.")
        else:
            print("Warning: total_num_entries is None, index check skipped.")

        return [(self.img_filepaths[index], self.lbl_filepaths[index])]

    def dump_as_tuple(self) -> tuple[list[str], list[str], int | None]:
        """Return the image and label file paths as a tuple."""
        return self.img_filepaths, self.lbl_filepaths, self.total_num_entries


def FetchDatasetPaths(dataset_name: str | list | tuple,
                      datasets_root_folder: str | tuple[str, ...],
                      samples_limit_per_dataset: int | tuple[int, ...] = 0):

    # Select loading mode (single or multiple datasets)
    if isinstance(dataset_name, str):
        dataset_names_array: list | tuple = (dataset_name,)

    elif isinstance(dataset_name, (list, tuple)):
        dataset_names_array: list | tuple = dataset_name
    else:
        raise TypeError(
            "dataset_name must be a string or a list of strings")

    # Initialize list index of datasets to load
    image_folder = []
    label_folder = []
    num_of_imags_in_set = []

    img_filepaths = []
    lbl_filepaths = []

    # Loop over index entries (1 per dataset folder to fetch)
    for dset_count, _dataset_name in enumerate(dataset_names_array):

        datasets_root_folder_ = None
        current_total_of_imgs = 0

        # Resolve correct root by check folder existence
        for root_count, datasets_root_folder_ in enumerate(datasets_root_folder):

            if os.path.exists(os.path.join(datasets_root_folder_, _dataset_name)):
                break

            elif root_count == len(datasets_root_folder) - 1:
                raise FileNotFoundError(
                    f"\033[91mDataset folder '{_dataset_name}' not found in any of the provided root folders: {datasets_root_folder}\033[0m")

        if datasets_root_folder_ is None:
            raise ValueError(
                f"\033[91mDataset folder cannot be None: automatic resolution of dataset root folder failed silently. Please check your configuration and report issue.\033[0m")

        print(
            f"Fetching dataset '{_dataset_name}' with root folder {datasets_root_folder_}...")

        # Append dataset paths
        image_folder.append(os.path.join(
            datasets_root_folder_, _dataset_name, "images"))

        label_folder.append(os.path.join(
            datasets_root_folder_, _dataset_name, "labels"))

        # Check size of names in the folder
        sample_file = next((f for f in os.listdir(image_folder[dset_count]) if os.path.isfile(
            os.path.join(image_folder[dset_count], f))), None)

        if sample_file:
            name_size = len(os.path.splitext(sample_file)[0])
            print(f"\tDataset name size: {name_size}. Example: {sample_file}")
        else:
            print("\033[38;5;208mNo files found in this folder!\033[0m")
            continue

        # Append number of images in the set
        num_of_imags_in_set.append(len(os.listdir(image_folder[dset_count])))

        # Build paths index
        if name_size == 6:
            img_filepaths.extend([os.path.join(
                image_folder[dset_count], f"{id+1:06d}.png") for id in range(num_of_imags_in_set[dset_count])])

        elif name_size == 8:
            img_filepaths.extend([os.path.join(
                image_folder[dset_count], f"{id*150:08d}.png") for id in range(num_of_imags_in_set[dset_count])])

        else:
            raise ValueError(
                "Image/labels names are assumed to have 6 or 8 numbers. Please check the dataset format.")

        # Check labels folder extensions
        file_ext = os.path.splitext(os.listdir(label_folder[dset_count])[0])[1]
        lbl_filepaths.extend([os.path.join(
            label_folder[dset_count], f"{id+1:06d}{file_ext}") for id in range(num_of_imags_in_set[dset_count])])

        current_total_of_imgs = sum(num_of_imags_in_set)  # Get current total
        print(
            f"Dataset '{_dataset_name}' contains {num_of_imags_in_set[dset_count]} images and labels in {file_ext} format.")

        # Check if samples limit is set and apply it if it does
        if isinstance(samples_limit_per_dataset, (list, tuple)):
            samples_limit_per_dataset_ = samples_limit_per_dataset[dset_count]
        else:
            samples_limit_per_dataset_ = samples_limit_per_dataset

        if samples_limit_per_dataset_ > 0:

            # Prune paths of the current dataset only
            img_filepaths = img_filepaths[current_total_of_imgs:
                                          current_total_of_imgs + samples_limit_per_dataset_]
            lbl_filepaths = lbl_filepaths[current_total_of_imgs:
                                          current_total_of_imgs + samples_limit_per_dataset_]

            print(
                f"\tLimiter: number of samples was limited to {samples_limit_per_dataset_}/{num_of_imags_in_set[dset_count]}")

            # Set total number of images to the limit
            num_of_imags_in_set[dset_count] = samples_limit_per_dataset_

    total_num_imgs = sum(num_of_imags_in_set)

    # Return paths container
    return DatasetPathsContainer(img_filepaths=img_filepaths,
                                 lbl_filepaths=lbl_filepaths,
                                 num_of_entries_in_set=num_of_imags_in_set,
                                 total_num_entries=total_num_imgs)

# %% Data containers
@dataclass
class ImagesLabelsContainer:
    """
    Container for storing images and their corresponding labels.

    Attributes:
        images (np.ndarray | torch.Tensor): Array or tensor containing image data.
        labels (np.ndarray | torch.Tensor): Array or tensor containing label data.
        labels_datakeys (tuple[PTAF_Datakey, ...] | None): Optional tuple specifying the data keys for the labels.
    """
    images: np.ndarray | torch.Tensor
    labels: np.ndarray | torch.Tensor

    labels_datakeys: tuple[PTAF_Datakey | str, ...] | None = None
    labels_sizes: dict[str, int] | None = None

    def __iter__(self):
        """
        Iterate over the images and labels.
        """
        for img, lbl in zip(self.images, self.labels):
            yield img, lbl


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


# %% Dataset base classes
class ImagesLabelsDatasetBase(Dataset):
    """
    A PyTorch Dataset for loading images with associated scene and camera metadata and labels.
    Supports 'pil' or 'cv2' backends for image loading.
    Scene metadata and labels are loaded per image; camera parameters once at init.
    """

    def __init__(self,
                 dset_cfg: ImagesDatasetConfig,
                 transform: Callable[[Image.Image], Any] | None = None,
                 ):
        """
        Args:
            image_paths: list of file paths to image metadata YAML files.
            camera_path: file path to camera parameters YAML file.
            transform: optional callable to apply to PIL Image samples.
        """
        super().__init__()
        # Store configuration
        self.dset_cfg = dset_cfg

        # Setup backend loader
        if self.dset_cfg.image_backend == 'cv2':
            try:
                import cv2
            except ImportError:
                raise ImportError(
                    "OpenCV (cv2) backend requested, but not installed")

            self._load_img_from_file: Callable = partial(cv2.imread,
                                                         flags=cv2.IMREAD_UNCHANGED)

        elif self.dset_cfg.image_backend == 'pil':
            try:
                from PIL import Image
            except ImportError:
                raise ImportError("PIL backend requested, but not installed")

            self._load_img_from_file = Image.open
        else:
            raise ValueError(
                f"Unsupported image_backend: {self.dset_cfg.image_backend}")

        # Store paths to images
        self.image_paths = image_paths

        if self.dset_cfg.camera_path is not None:
            # Load camera parameters
            self.camera_params = self._load_yaml(self.dset_cfg.camera_path)
        else:
            # Initialize camera parameters as empty dict
            self.camera_params = {}

        # Store transform function
        # TODO input is a PIL image, but may not be the best format to load
        self.transform = transform

        # Cache attribute for processed labels
        self._label_cache: dict[str, Any] = {}

        # Build paths index
        self.dataset_paths_container = FetchDatasetPaths(
            dataset_name=self.dset_cfg.dataset_names_list,
            datasets_root_folder=self.dset_cfg.datasets_root_folder,
            samples_limit_per_dataset=self.dset_cfg.samples_limit_per_dataset
        )

        self.dataset_size = len(self.image_paths)

    def _load_yaml(self, path: str) -> dict[str, Any]:
        """
        Load and return YAML content as dict.
        """
        pass  # DEVNOTE what purpose?

    def __len__(self) -> int:
        return self.dataset_size

    # TODO review/rework
    def _load_image(self, img_path: str) -> dict[str, Any]:
        """
        Load image via selected backend.
        """

        # Image loading (call backend method)
        img = self._load_img_from_file(img_path)

        if img is None:
            raise FileNotFoundError(
                f"Failed to load image {img_path} with backend {self.image_backend}"
            )

        return self._scale_image(img)

    def _scale_image(self, img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Scale image data based on dtype or custom scaling.
        """

        if not isinstance(img, (np.ndarray, torch.Tensor)):
            raise TypeError("Image must be a numpy array or a torch tensor.")
        
        if self.dset_cfg.intensity_scaling_mode not in ['none', 'dtype', 'custom']:
            raise ValueError(
                f"Unsupported intensity scaling mode: {self.dset_cfg.intensity_scaling_mode}")

        if self.dset_cfg.intensity_scaling_mode == 'none':
            return img

        elif self.dset_cfg.intensity_scaling_mode == 'dtype':
            # Scale based on dtype
            if isinstance(img, np.ndarray):
                if img.dtype == np.uint8:
                    return img.astype(np.float32) / 255.0
                elif img.dtype == np.uint16:
                    return img.astype(np.float32) / 65535.0
                else:
                    raise TypeError(
                        "Unsupported image data type for scaling. Only uint8 and uint16 are supported.")
                
            elif isinstance(img, torch.Tensor):
                if img.dtype == torch.uint8:
                    return img.float() / 255.0
                elif img.dtype == torch.uint16:
                    return img.float() / 65535.0
                else:
                    raise TypeError(
                        "Unsupported image tensor data type for scaling. Only uint8 and uint16 are supported.")
            else:
                raise TypeError("Image must be a numpy array or a torch tensor.")

        elif self.dset_cfg.intensity_scaling_mode == 'custom':
            if self.dset_cfg.intensity_scale_value is None:
                raise ValueError("intensity_scale_value must be set when intensity_scaling_mode is 'custom'.")
            
            return img * self.dset_cfg.intensity_scale_value


    # TODO review/rework
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

    def __getitem__(self, idx: int):
        # Check index is not out of bounds
        if idx < 0 or idx >= self.dataset_size:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {self.dataset_size}")

        # Get paths
        image_path, label_path = self.dataset_paths_container[idx]

        # Load image
        img = torch.tensor(self._load_image(image_path), 
                           dtype=self.dset_cfg.image_dtype)

        # Load labels from YAML file
        lbl = LabelsContainer.load_from_yaml(label_path)

        # Extract lbl data based on datakeys
        lbl = torch.tensor(lbl.get_labels(data_keys=self.dset_cfg.lbl_vector_data_keys),
                           dtype=self.dset_cfg.lbl_dtype)

        if self.transform is not None:
            # Apply transform to the image and label
            img = self.transform(img)

        if self.target_transform is not None:
            # Apply target transform to the label
            lbl = self.target_transform(lbl)

        # Load image and labels
        return img, lbl

    def _process_labels(self, raw_labels: dict[str | PTAF_Datakey, Any]) -> Any:
        """
        Placeholder for label processing. Override in subclass.
        """
        return raw_labels

    # DEVNOTE: this method is not up to date
    # TODO reevaluate need and redesign
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

        pattern = os.path.join(root_dir,  f"*{image_meta_ext}")
        image_paths = sorted(glob.glob(pattern))
        camera_path = os.path.join(root_dir, camera_meta_filename)

        return cls(image_paths, camera_path, transform, image_backend)


class ImagesLabelsCachedDataset(TensorDataset, ImagesLabelsDatasetBase):
    """
    A cached dataset for images and labels, inheriting from both TensorDataset and ImagesLabelsDatasetBase.

    This class allows efficient loading of pre-cached images and labels, supporting optional transformations.
    It expects either an ImagesLabelsContainer or paths to images and labels (not implemented yet).

    Attributes:
        input_scale_factor (float): Factor to normalize image data based on dtype.
        transforms (torch.nn.Module | Compose | None): Optional transformations to apply to images and labels.

    Args:
        images_labels (ImagesLabelsContainer | None): Container holding images and labels as tensors or arrays.
        transforms (torch.nn.Module | Compose | None): Optional transformations to apply to images and labels.
        images_path (str | None): Path to images file (not implemented).
        labels_path (str | None): Path to labels file (not implemented).

    Raises:
        TypeError: If images_labels is not an ImagesLabelsContainer.
        ValueError: If neither images_labels nor both images_path and labels_path are provided.
        NotImplementedError: If loading from paths is attempted.
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

    def _process_labels(self, raw_labels: dict[str, Any]) -> Any:
        raise NotImplementedError("Method to implement")


class TupledImagesLabelsCachedDataset(ImagesLabelsDatasetBase):
    def __init__(self, tupled_images_labels: TupledImagesLabelsContainer):
        """
        TupledImagesLabelsCachedDataset is a dataset class for cases where each input is a tuple,
        such as (image, vector), paired with corresponding labels.

        Args:
            tupled_images_labels (TupledImagesLabelsContainer): Container holding a tuple of input arrays/tensors and labels.
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

# %% Dataset classes for specialized formats


class ImagesDataset_StandardESA(ImagesLabelsDatasetBase):
    """
    A PyTorch Dataset for loading images with associated scene and camera metadata and labels.
    Scene metadata (Table 1) and labels are loaded per image; camera parameters (Table 2)
    are loaded once at initialization.
    """

    def __init__(
        self,
        image_meta_paths: list[str],
        camera_meta_path: str,
        transform: Callable[[Image.Image], Any] | None = None
    ):
        """
        Args:
            image_meta_paths: list of file paths to image metadata YAML files.
            camera_meta_path: file path to camera parameters YAML file.
            transform: optional callable to apply to PIL Image samples.
        """
        super().__init__()
        self.image_meta_paths = image_meta_paths
        self.camera_params = self._load_yaml(camera_meta_path)
        self.transform = transform

        # Cache for processed labels
        self._label_cache: dict[str, Any] = {}

    def _load_yaml(self, path: str) -> dict[str, Any]:
        """
        Load and return YAML content as dict.
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def __len__(self) -> int:
        return len(self.image_meta_paths)

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], Any]:
        """
        Return (sample, labels) for training.
        """
        meta_path = self.image_meta_paths[idx]
        sample = self.load_single(meta_path)
        labels = self.load_labels(meta_path)
        return sample, labels

    def load_single(self, image_meta_path: str) -> dict[str, Any]:
        """
        Load image, scene metadata, and camera parameters for one sample.
        """
        scene_meta = self._load_yaml(image_meta_path)
        img_filename = scene_meta.get('image')

        if img_filename is None:
            raise KeyError(f"'image' key not found in {image_meta_path}")
        img_path = os.path.join(os.path.dirname(image_meta_path), img_filename)

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'scene_metadata': scene_meta,
            'camera_parameters': self.camera_params
        }

    def _process_labels(self, raw_labels: dict[str, Any]) -> Any:
        """
        Placeholder for label processing. Override in subclass.
        """
        return raw_labels


# %% LEGACY CODE
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
