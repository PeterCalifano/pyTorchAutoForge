from ast import Import
import enum
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import sys
from dataclasses import dataclass, field
import copy

from torch.utils.data.dataset import TensorDataset
from pathlib import Path
from pyTorchAutoForge.utils import numpy_to_torch, Align_batch_dim, torch_to_numpy
from torchvision.transforms import Compose

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from abc import ABC, abstractmethod
from abc import ABCMeta
from typing import Any, Literal, TYPE_CHECKING
from collections.abc import Callable
import yaml
from PIL import Image
from functools import partial
import json
import yaml
from pyTorchAutoForge.datasets.LabelsClasses import PTAF_Datakey, LabelsContainer

try:
    import cv2 as ocv
    hasOpenCV: bool = True

except ImportError:
    hasOpenCV: bool = False

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
    dataset_names_list: Path | str | list[str | Path] | tuple[str | Path, ...]
    datasets_root_folder: Path | str | tuple[str | Path, ...]
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
        if isinstance(self.datasets_root_folder, (str, Path)):
            self.datasets_root_folder = (self.datasets_root_folder,)
        if isinstance(self.dataset_names_list, (str, Path)):
            self.dataset_names_list = (self.dataset_names_list,)

        # Check all roots exist
        for root in self.datasets_root_folder:
            if not os.path.exists(root):
                raise FileNotFoundError(
                    f"Dataset root folder '{root}' does not exist.")

        # Check all datakeys are valid PTAF_Datakey
        lbl_vector_data_keys_checked = []
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

            lbl_vector_data_keys_checked.append(key)

        # Reassign lbl_vector_data_keys to ensure they are PTAF_Datakey instances
        self.lbl_vector_data_keys = tuple(lbl_vector_data_keys_checked)


@dataclass
class ImagesDatasetConfig(DatasetLoaderConfig):
    # Default options
    binary_masks_folder_name: str = "binary_masks"
    images_folder_name: str = "images"
    camera_filepath: str | None = None

    image_format: str = "png"
    image_dtype: type | torch.dtype = torch.uint8
    image_backend: Literal['pil', 'cv2'] = 'cv2'
    load_as_tensor: bool = True

    # Additional options
    intensity_scaling_mode: Literal['none', 'dtype', 'custom'] = 'dtype'
    # Used only if intensity_scaling is 'custom'
    intensity_scale_value: float | None = None
    output_size_limit: int = -1

    def __post_init__(self):
        super().__post_init__()

        if self.intensity_scaling_mode not in ['none', 'dtype', 'custom']:
            raise ValueError(
                f"Unsupported intensity scaling mode: {self.intensity_scaling_mode}. Supported modes are 'none', 'dtype', and 'custom'.")
        else:
            print('Selected intensity scaling mode:',
                  self.intensity_scaling_mode)

        if self.intensity_scale_value is not None and self.intensity_scaling_mode != 'custom':
            raise ValueError(
                f"intensity_scale_value must be None unless intensity_scaling_mode is 'custom'.")


######################## DEVNOTE Relatively stable code BELOW ########################
@dataclass
class DatasetPathsContainer():
    """
    Container for storing and accessing image and label file paths.

    This class manages collections of paired image and label file paths for one or more
    datasets. It ensures the paths are properly matched and provides indexed access to
    retrieve image-label path pairs.

    Attributes:
        img_filepaths: List of file paths to images.
        lbl_filepaths: List of file paths to corresponding labels.
        total_num_entries: Total number of image-label pairs across all datasets.
        num_of_entries_in_set: Number of entries in each dataset, as list or single int.

    Raises:
        ValueError: If image and label file paths are None or don't have matching lengths.
        IndexError: If an index is out of bounds when accessing items.
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

    def __getitem__(self, index: int | list[int]) -> list[tuple[str, str]] | tuple[str, str]:
        """Get the image and label file paths for a given index."""

        if isinstance(index, (list, tuple)):
            # Return list of tuple pairs [(img_i, lbl_i)]
            return [(self.img_filepaths[i], self.lbl_filepaths[i]) for i in index]

        if self.total_num_entries is not None:
            if index < 0 or index >= self.total_num_entries:
                raise IndexError("Index out of range.")
        else:
            print("Warning: total_num_entries is None, index check skipped.")

        # Return tuple pair (img, lbl)
        return self.img_filepaths[index], self.lbl_filepaths[index]

    def dump_as_tuple(self) -> tuple[list[str], list[str], int | None]:
        """Return the image and label file paths as a tuple."""
        return self.img_filepaths, self.lbl_filepaths, self.total_num_entries


@dataclass
class SamplesSelectionCriteria():
    max_apparent_size: float | int | None = None
    min_bbox_width_height: tuple[float,
                                 float] | tuple[int, int] | float | int | None = None
    min_Q75_intensity: int | float | None = None

    def __post_init__(self):
        # Handle scalars to tuple
        if isinstance(self.min_bbox_width_height, (float, int)):
            self.min_bbox_width_height = (
                self.min_bbox_width_height, self.min_bbox_width_height)

        # Check type and values validity
        if not isinstance(self.min_bbox_width_height, (tuple, list)) and \
                not self.min_bbox_width_height is None:
            raise TypeError(
                "min_bbox_width_height must be a tuple, list, float, or int.")
        elif self.min_bbox_width_height is not None:
            # Check size
            if len(self.min_bbox_width_height) != 2:
                raise ValueError(
                    "min_bbox_width_height must be a tuple or list of two elements.")

            if not self.min_bbox_width_height[0] >= 0.0 and self.min_bbox_width_height[1] >= 0.0:
                raise ValueError(
                    "min_bbox_width_height values must be non-negative.")

        if self.min_Q75_intensity is not None and self.min_Q75_intensity < 0:
            raise ValueError(
                "min_Q75_intensity must be a non-negative value.")

        if self.max_apparent_size is not None and self.max_apparent_size < 0:
            raise ValueError("max_apparent_size must be a non-negative value.")

    def is_valid_img_lbl(self, img_path, lbl_path) -> bool:
        # Function to check whether (img, lbl) pair is a valid one based on selection criteria
        # TODO copy and adapt code from FetchDatasetPaths
        raise NotImplementedError('TODO')
        return False


def FetchDatasetPaths(dataset_name: Path | str | list[str | Path] | tuple[str | Path, ...],
                      datasets_root_folder: Path | str | tuple[str | Path, ...],
                      samples_limit_per_dataset: int | tuple[int, ...] = 0,
                      selection_criteria: SamplesSelectionCriteria | None = None) -> DatasetPathsContainer:
    """Fetches file paths for images and labels from specified datasets.

    Locates and builds paths to image and label files from one or more datasets,
    handling various naming conventions and optional sample limiting.

    Args:
        dataset_name: Name(s) of dataset folder(s) to fetch paths from. Can be a single
            string/Path or a collection of strings/Paths.
        datasets_root_folder: Root folder(s) containing the dataset folders. Can be a 
            single string/Path or a collection of strings/Paths.
        samples_limit_per_dataset: Maximum number of samples to include from each dataset.
            If > 0, limits the dataset size. Can be a single integer applied to all datasets
            or a collection of integers with one limit per dataset. Defaults to 0 (no limit).

    Raises:
        TypeError: If dataset_name is not a string or a list/tuple of strings.
        FileNotFoundError: If a dataset folder is not found in any of the provided root folders.
        ValueError: If automatic resolution of dataset root folder fails or if image/label 
            naming convention is not supported.

    Returns:
        DatasetPathsContainer: Container with paths to images and labels, along with dataset size info.
    """

    # Select loading mode (single or multiple datasets)
    if isinstance(dataset_name, (str, Path)):
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

        # Build filepath template
        if name_size == 6:
            filepath_template = lambda id: f"{id+1:06d}"

        elif name_size == 8:
            filepath_template = lambda id: f"{id*150:08d}"

        else:
            raise ValueError(
                "Image/labels names are assumed to have 6 or 8 numbers. Please check the dataset format.")

        # Get labels folder extensions
        file_ext = os.path.splitext(
            os.listdir(label_folder[dset_count])[0])[1]

        # Fetch sample paths and select if required
        if selection_criteria is not None:
            # Build temporary path index and select samples
            tmp_img_count = num_of_imags_in_set[dset_count]

            # Build temporary numpy array of chars of size equal to num_of_imags_in_set[dset_count]
            filenames = [filepath_template(id) + ".png" for id in range(tmp_img_count)]
            tmp_img_filepaths = [os.path.join(image_folder[dset_count], path) for path in filenames]

            # Pick a safe fixed-width dtype to avoid object arrays
            max_path_len = max(len(p) for p in tmp_img_filepaths)
            tmp_img_filepaths = np.array(tmp_img_filepaths, dtype=f"U{max_path_len}")

            # Define lbl filepaths and mask arrays
            tmp_lbl_filepaths = np.empty(tmp_img_count, dtype=f"U{max_path_len}") 
            tmp_valid_mask = np.zeros(tmp_img_count, dtype=bool)

            # DEVNOTE: can be parallelized!
            for id in range(num_of_imags_in_set[dset_count]):
                # Build image path
                tmp_img_path = os.path.join(
                    image_folder[dset_count], filepath_template(id) + '.png')

                # If any selection is requested based on image intensity, load the image
                if selection_criteria.min_Q75_intensity is not None and hasOpenCV:
                    # Load image
                    img = ocv.imread(tmp_img_path, ocv.IMREAD_UNCHANGED)
                    image_scaling_coeff = 1.0

                    if img.dtype == 'uint8':
                        image_scaling_coeff = 1.0

                        # Set all 1.0 to 0.0 to fix Blender images issue
                        img[img == 1.0] = 0.0

                    elif img.dtype == 'uint16':
                        image_scaling_coeff = 1.0/(257.01)

                        # Set all 1.0 to 0.0 to fix Blender images issue
                        img[img == 1.0] = 0.0

                    elif img.dtype == 'uint8':
                        # Images from ABRAM have 12-bit depth?
                        image_scaling_coeff = 1.0

                    elif img.dtype != 'uint8' and img.dtype != 'uint16':
                        raise TypeError(
                            "Image data type is neither uint8 nor uint16. This dataset loader only supports these two data types.")

                    # Compute percentile of intensity considering non-zero pixels
                    scaled_img = image_scaling_coeff * img.astype(np.float32)
                    nonzero_pixels = scaled_img[scaled_img > 0]

                    perc75_intensity = np.percentile(
                        nonzero_pixels, 75) if nonzero_pixels.size > 0 else 0
                    if perc75_intensity < 5 or \
                        (perc75_intensity < selection_criteria.min_Q75_intensity and
                         sum(nonzero_pixels) < 2E3) or \
                            (perc75_intensity < 1.2 * selection_criteria.min_Q75_intensity and \
                             sum(nonzero_pixels) < 5E3):
                        
                        print(f" - Warning: Image {id+1} has 75th percentile intensity of illuminated pixels equal to {perc75_intensity} < {selection_criteria.min_Q75_intensity} in too many pixels, discarded as too dark.")
                        continue
                elif selection_criteria.min_Q75_intensity is not None and not hasOpenCV:
                    print(f"\033[38;5;208mWARNING: OpenCV is not installed, cannot check image intensity. Skipping selection based on min_Q75_intensity.\033[0m", end='\r')


                # Build lbl path
                tmp_lbl_path = os.path.join(
                    label_folder[dset_count], filepath_template(id) + file_ext)

                if os.path.exists(tmp_lbl_path):
                    if tmp_lbl_path.endswith('.yml') or tmp_lbl_path.endswith('.yaml'):
                        labelFile = LabelsContainer.load_from_yaml(tmp_lbl_path)
                    else:
                        raise ValueError(
                            f"Unsupported label file format: {tmp_lbl_path}. Only .yml and .yaml are supported byt this implementation.")
                else:
                    raise FileNotFoundError(f"Label file not found: {tmp_lbl_path}")
                
                # Add in array
                if selection_criteria.max_apparent_size is not None:
                    lbl_check_val = labelFile.get_labels(
                        data_keys=PTAF_Datakey.APPARENT_SIZE)

                    if lbl_check_val is not None and lbl_check_val[0] > selection_criteria.max_apparent_size:
                        print(f" - Warning: Image {id+1} has apparent size {lbl_check_val[0]} > {selection_criteria.max_apparent_size}, discarded as too large.")
                        continue

                # If discard based on bounding box size is requested check it
                if selection_criteria.min_bbox_width_height is not None:
                    min_bbox_width_height = selection_criteria.min_bbox_width_height
                    if isinstance(min_bbox_width_height, (float, int)):
                        min_bbox_width_height = (min_bbox_width_height, min_bbox_width_height)

                    lbl_check_val = labelFile.get_labels(
                        data_keys=PTAF_Datakey.BBOX_XYWH)

                    if lbl_check_val is not None and lbl_check_val[2] < min_bbox_width_height[0] and \
                        lbl_check_val[3] < min_bbox_width_height[1]:
                        print(f" - Warning: Image {id+1} has bounding box width, height = {lbl_check_val[2:4]} < {min_bbox_width_height}, discarded as too small.")
                        continue

                # Mark sample as valid if all checks are GO
                tmp_valid_mask[id] = True
                tmp_lbl_filepaths[id] = tmp_lbl_path
                tmp_img_filepaths[id] = tmp_img_path

            # Extract valid samples
            tmp_img_filepaths = tmp_img_filepaths[tmp_valid_mask].tolist()
            tmp_lbl_filepaths = tmp_lbl_filepaths[tmp_valid_mask].tolist()

            # Update count of images
            prev_size = num_of_imags_in_set[dset_count]
            num_of_imags_in_set[dset_count] = len(tmp_img_filepaths)

        else:
            # Build paths WITHOUT selection based on samples content
                      
            # Get all paths as before
            tmp_img_filepaths = [os.path.join(image_folder[dset_count], filepath_template(
                id) + ".png") for id in range(num_of_imags_in_set[dset_count])]

            tmp_lbl_filepaths = [os.path.join(label_folder[dset_count], filepath_template(
                id) + file_ext) for id in range(num_of_imags_in_set[dset_count])]

            prev_size = num_of_imags_in_set[dset_count]

        # Append paths to filepaths lists
        img_filepaths.extend(tmp_img_filepaths)
        lbl_filepaths.extend(tmp_lbl_filepaths)

        current_total_of_imgs = sum(num_of_imags_in_set)  # Get current total
        print(f"Dataset '{_dataset_name}' contains {num_of_imags_in_set[dset_count]} valid images and labels in {file_ext} format. Removed by selection criteria: {prev_size - num_of_imags_in_set[dset_count]}.")

        # Check if samples limit is set and apply it if it does
        if isinstance(samples_limit_per_dataset, (list, tuple)):
            samples_limit_per_dataset_ = samples_limit_per_dataset[dset_count]
        else:
            samples_limit_per_dataset_ = samples_limit_per_dataset

        if samples_limit_per_dataset_ > 0:

            # Prune paths of the current dataset only
            img_filepaths = img_filepaths[current_total_of_imgs - num_of_imags_in_set[dset_count]:
                                          current_total_of_imgs - num_of_imags_in_set[dset_count] + samples_limit_per_dataset_]
            lbl_filepaths = lbl_filepaths[current_total_of_imgs - num_of_imags_in_set[dset_count]:
                                          current_total_of_imgs - num_of_imags_in_set[dset_count] + samples_limit_per_dataset_]

            print(
                f"\tLIMITER: number of samples was limited to {samples_limit_per_dataset_}/{num_of_imags_in_set[dset_count]}")

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

    def __copy__(self):
        """
        Create a deep copy of the container.
        """
        return ImagesLabelsContainer(
            images=copy.deepcopy(self.images),
            labels=copy.deepcopy(self.labels),
            labels_datakeys=copy.deepcopy(self.labels_datakeys),
            labels_sizes=copy.deepcopy(self.labels_sizes)
        )


# TODO Update/remove
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
    def __init__(self,
                 dset_cfg: ImagesDatasetConfig,
                 transform: torch.nn.Module | None = None,
                 lbl_transform: torch.nn.Module | None = None,
                 skip_output_slicing: bool = False,
                 selection_criteria: SamplesSelectionCriteria | None = None):
        """
        Initialize a base dataset for images and labels.

        Sets up the dataset configuration, image loading backend, and transforms for 
        both images and labels. Fetches dataset paths and prepares for data loading.

        Args:
            dset_cfg: Configuration object containing dataset paths, file formats, 
                and loading parameters.
            transform: Optional transformation module to apply to loaded images.
            lbl_transform: Optional transformation module to apply to loaded labels.

        Raises:
            ImportError: If the requested image backend (cv2 or PIL) is not installed.
            ValueError: If an unsupported image backend is specified.
        """
        super().__init__()
        # Store configuration
        self.dset_cfg = dset_cfg
        self.skip_output_slicing = skip_output_slicing

        if selection_criteria is not None and not isinstance(selection_criteria, SamplesSelectionCriteria):
            raise TypeError(f'selection_criteria must be either None or a SamplesSelectionCriteria instance. Got {type(selection_criteria)}.')
        
        self.selection_criteria = selection_criteria

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
        if self.dset_cfg.camera_filepath is not None:
            # Load camera parameters
            self.camera_params = self._load_yaml(self.dset_cfg.camera_filepath)
        else:
            # Initialize camera parameters as empty dict
            self.camera_params = {}

        # Store transform function
        # TODO input is a PIL image, but may not be the best format to load
        self.transform = transform
        self.lbl_transform = lbl_transform

        # Cache attribute for processed labels
        self._label_cache: dict[str, Any] = {}

        # Build paths index
        self.dataset_paths_container = FetchDatasetPaths(dataset_name=self.dset_cfg.dataset_names_list,
                                                         datasets_root_folder=self.dset_cfg.datasets_root_folder,
                                                         samples_limit_per_dataset=self.dset_cfg.samples_limit_per_dataset,
                                                         selection_criteria=self.selection_criteria)

        self.dataset_size = len(self.dataset_paths_container)

        # TODO add code to determine input scale factor based on dtype
        # self.input_scale_factor

    def _load_yaml(self, path: str) -> dict[str, Any]:
        """
        Load and return YAML content as dict.
        """
        with open(path, 'r') as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to load YAML file {path}: {e}")

    def __len__(self) -> int:
        return self.dataset_size

    def _load_image(self, img_path: str) -> np.ndarray | torch.Tensor:
        """
        Load image via selected backend.
        """

        # Image loading (call backend method)
        img = self._load_img_from_file(img_path)

        if img is None:
            raise FileNotFoundError(
                f"Failed to load image {img_path} with backend {self.dset_cfg.image_backend}"
            )

        if self.dset_cfg.load_as_tensor:
            img = numpy_to_torch(img)

            if len(img.shape) == 2:
                img = img.unsqueeze(0)  # Unsqueeze from (H,W) to (C,H,W))

            elif img.shape[-1] <= 3:
                # Convert to (C,H,W) format
                img = img.permute(2, 0, 1)

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
                raise TypeError(
                    "Image must be a numpy array or a torch tensor.")

        elif self.dset_cfg.intensity_scaling_mode == 'custom':
            if self.dset_cfg.intensity_scale_value is None:
                raise ValueError(
                    "intensity_scale_value must be set when intensity_scaling_mode is 'custom'.")

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
        img = numpy_to_torch(self._load_image(image_path))  # type:ignore

        # Load labels from YAML file
        lbl = LabelsContainer.load_from_yaml(label_path)

        # Extract lbl data based on datakeys
        lbl = torch.tensor(lbl.get_labels(data_keys=self.dset_cfg.lbl_vector_data_keys),
                           dtype=self.dset_cfg.lbl_dtype)  # type:ignore

        if self.transform is not None:
            # Apply transform to the image and label
            img = self.transform(img)

        if self.lbl_transform is not None:
            # Apply target transform to the label
            lbl = self.lbl_transform(lbl)

        # Slice lbl if output_size_limit is set
        if self.dset_cfg.output_size_limit > 0 and not (self.skip_output_slicing):
            if lbl.shape[0] > self.dset_cfg.output_size_limit:
                lbl = lbl[:self.dset_cfg.output_size_limit]

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

    def get_all_labels_container(self):
        """
        Get the labels container for this dataset.

        Returns:
            LabelsContainer: The labels container with the specified data keys.
        """

        # Loop over paths and get labels vectors
        lbl_vector_size, lbl_size_dict = LabelsContainer.get_lbl_1d_vector_size(
            data_keys=self.dset_cfg.lbl_vector_data_keys)

        lbl_array = np.zeros(
            (len(self.dataset_paths_container.lbl_filepaths), lbl_vector_size))

        for id_lbl, lbl_path in enumerate(self.dataset_paths_container.lbl_filepaths):
            print(
                f"Fetching labels from disk: {id_lbl + 1}/{len(self.dataset_paths_container.lbl_filepaths)}", end='\r')
            # Load labels from YAML file
            lbl = LabelsContainer.load_from_yaml(lbl_path)

            # Extract lbl data based on datakeys
            lbl = lbl.get_labels(data_keys=self.dset_cfg.lbl_vector_data_keys)
            lbl_array[id_lbl] = np.array(lbl)

        container = ImagesLabelsContainer(images=np.empty_like((0, 0)),
                                          labels=lbl_array,
                                          labels_datakeys=self.dset_cfg.lbl_dtype,
                                          labels_sizes=lbl_size_dict)
        return container


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
