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

# DEVNOTE (PC) this is an attempt to define a configuration class that allows a user to specify dataset structure to drive the loader, in order to ease the use of diverse dataset formats

# %% Types and aliases
class ImagesDatasetType(enum.Enum):
    """
    Enumeration class for dataset types.
    """
    SEQUENCED = "ScatteredSequences"
    POINT_CLOUD = "point_cloud" # TODO modify
    TRAJECTORY = "trajectory" # TODO modify

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

# %% Configuration classes
@dataclass
class DatasetLoaderConfig():
    """
    DatasetStructureConfig _summary_

    _extended_summary_
    """
    # Required fields
    # Generics
    # DEVNOTE expand to also accept a list of names under the same root for "automerging"?
    dataset_names: Path | str | list[str | Path]

    # Labels
    label_size: int = 0  # DEVNOTE why needed?
    num_samples: int = 0
    label_folder_name: str = ""

    # Optional
    # Generics
    dataset_root_path: str = "."
    hostname: str = "localhost"  # Default is local machine

    # Additional details/options
    max_num_samples: int = -1  # If -1, all samples are loaded

    def __post_init__(self):
        """
        Post-initialization checks for the DatasetLoaderConfig.
        """
        # TODO add code to define dataset_root_path from dataset_name if not provided (de)


@dataclass
class ImagesDatasetConfig(DatasetLoaderConfig):
    image_folder: str = ""
    image_format: str = "png"
    image_dtype: type | torch.dtype = np.uint8

######################## DEVNOTE Relatively stable code BELOW ########################
# %% Path/data fetching functions
# TODO take from ML-Gears



# %% Data containers
@dataclass
class ImagesLabelsContainer:
    """
     _summary_

    _extended_summary_
    """
    images: np.ndarray | torch.Tensor
    labels: np.ndarray | torch.Tensor

    def __iter__(self):
        """
        Iterate over the images and labels.
        """
        for img, lbl in zip(self.images, self.labels):
            yield img, lbl
    
    def __getitem__(self, idx):
        """
        Get the image and label at the specified index.
        """
        if isinstance(idx, slice):
            return ImagesLabelsContainer(
                images=self.images[idx],
                labels=self.labels[idx]
            )
        else:
            return self.images[idx], self.labels[idx]

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

# %% Dataset format classes
# DEVNOTE format_types group all available formats supported by the dataset index class. May be changed to a registry?
dataset_format_types = Literal["ImagesDatasetFormat_Sequences",
                               "ImagesDatasetFormat_PointCloud",
                               "ImagesDatasetFormat_Trajectory"]

@dataclass
class ImagesDatasetFormat(ABC):

    @property
    @abstractmethod
    def dataset_type(self) -> ImagesDatasetType:
        pass

    @property
    @abstractmethod
    def collection_name(self) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def __str__(self):
        # Get name of the type
        type_name : dataset_format_types = self.__class__.__name__
        return type_name

    target_object: str
    dataset_id: int


class ImagesDatasetFormat_Sequences(ImagesDatasetFormat):
    """
    Class to specify sequenced datasets config format.
    """
    num_sequences: int

    def __init__(self, target_object: str, dataset_id: int, num_sequences: int):
        super().__init__(target_object, dataset_id)
        self.num_sequences = num_sequences

    @property
    def dataset_type(self) -> ImagesDatasetType:
        return ImagesDatasetType.SEQUENCED

    @property
    def collection_name(self) -> str:
        return "UniformlyScatteredSequencesDatasets"
    
    def get_name(self) -> str:
        return f"Dataset_{self.dataset_type.value}_{self.target_object}_{self.num_sequences}seqs_ID{self.dataset_id}"
    
class ImagesDatasetFormat_PointCloud(ImagesDatasetFormat):
    """
    Class to specify point cloud datasets config format.
    """
    def __init__(self, target_object: str, dataset_id: int):
        super().__init__(target_object, dataset_id)

    @property
    def dataset_type(self) -> ImagesDatasetType:
        return ImagesDatasetType.POINT_CLOUD

    @property
    def collection_name(self) -> str:
        return "UniformlyScatteredPointCloudsDatasets"
    
    def get_name(self) -> str:
        raise NotImplementedError("get_name() not implemented yet for ImagesDatasetFormat_PointCloud")

class ImagesDatasetFormat_Trajectory(ImagesDatasetFormat):
    """
    Class to specify trajectory datasets config format.
    """
    def __init__(self, target_object: str, dataset_id: int):
        super().__init__(target_object, dataset_id)

    @property
    def dataset_type(self) -> ImagesDatasetType:
        return ImagesDatasetType.TRAJECTORY

    @property
    def collection_name(self) -> str:
        return "TrajectoriesDatasets"

    def get_name(self) -> str:
        raise NotImplementedError("get_name() not implemented yet for ImagesDatasetFormat_Trajectory")
    

# %% Dataset index classes
# TODO
@dataclass
class DatasetIndex:
    dataset_root_path: Path | str | None = None
    dataset_format_objects: dataset_format_types | None = None
    dataset_name: str | None = None

    dataset_inputs_paths : list[str | Path] = field(default_factory=list)
    dataset_targets_paths : list[str | Path] = field(default_factory=list)

    # Optional settings
    img_name_folder: str = "images"
    label_name_folder: str = "labels"
    events_name_folder : str = "events"
    masks_name_folder: str = "masks"
    visibility_masks_name_folder: str = "binary_masks"

    def __post_init__(self):
        # Build index
        if self.dataset_root_path is None:
            print("Dataset root path not provided. Index cannot be built.")
            return
        

    @classmethod
    def load(cls):
        raise NotImplementedError(
            "DatasetIndex.load() is not implemented yet. Please implement this method to load dataset index from a file or other source.")
    
class DatasetsIndexTree(dict):
    def __init__(self,
                 dataset_root_path: Path | str | list[Path | str] | None = None,
                 dataset_format_objects: ImagesDatasetFormat | list[ImagesDatasetFormat] = None):
        """
        Initialize the DatasetsIndex with a root path and dataset format objects.

        Args:
            dataset_root_path (Path | str | list[Path | str] | None): Root path for datasets.
            dataset_format_objects (ImagesDatasetFormat | list[ImagesDatasetFormat]): Dataset format objects.
        """
        super().__init__()

        # Initialize attributes
        self.dataset_root_paths = dataset_root_path
        self.dataset_format_objects = dataset_format_objects 
        self.dataset_names = []
        self.dataset_paths = []

        self.dataset_indices: list[DatasetIndex] = []

    def __post_init__(self):
        """
        __post_init__ _summary_

        _extended_summary_
        """

        # TODO move code in a dedicated method such that it can be reused for __append__

        # For each dataset format object in the list:
        # - build the dataset name
        # - check it exists
        # - get the collection name
        # - assign it a unique id

        # Convert all strings to path objects
        if self.dataset_root_path is None:
            print("Dataset root path not provided.")
            return
        
        if isinstance(self.dataset_root_path, str):
            self.dataset_root_path = Path(self.dataset_root_path)

        if not self.dataset_root_path.exists():
            print(f"Dataset root path {self.dataset_root_path} does not exist.")

        # Wrap into tuple if not already
        if isinstance(self.dataset_format_objects, ImagesDatasetFormat):
            self.dataset_format_objects = [self.dataset_format_objects]

        self.dataset_names = []
        collection_names = []
        self.dataset_paths = []
        idx = 0

        for dset_format_ in self.dataset_format_objects:

            name_ = Path(dset_format_.get_name())
            target_name_ = Path(dset_format_.target_object)
            collection_name_ = Path(dset_format_.collection_name)

            full_path_ = self.dataset_root_path / collection_name_ / target_name_ / name_

            # Check the dataset exists
            if not (full_path_).exists():
                print(f"Dataset path '{full_path_}' does not exist.")
                continue

            # Append and increment index if existing
            self.dataset_names.append(name_)
            collection_names.append(collection_name_)
            idx += 1

            # Build dataset path
            self.dataset_paths.append(full_path_)
        
    def __call__(self, index):
        """
        Get the dataset path at the given index.
        """
        if index < 0 or index >= len(self.dataset_indices):
            raise IndexError("Index out of range for dataset indices.")

        return self.dataset_indices[index]

    def __str__(self):
        """
        String representation of the dataset index.
        """
        result = ["Datasets Index:"]
        for i, (name, path) in enumerate(zip(self.dataset_names, self.dataset_paths)):
            result.append(f" Key {i}:\n\tName: {name}\n\tPath: {path}")
        return "\n".join(result)

    def __append__(self, dataset_format_object: ImagesDatasetFormat):
        """
        Append a new dataset to the index providing its data.
        """
        if not isinstance(dataset_format_object, ImagesDatasetFormat):
            raise TypeError("dataset_format_object must be an instance of ImagesDatasetFormat.")

        if isinstance(self.dataset_format_objects, list):

            # Check if the dataset format object already exists
            if dataset_format_object in self.dataset_format_objects:
                print(f"Dataset format object {dataset_format_object} already exists in the index.")
                return
            
            self.dataset_format_objects.append(dataset_format_object)

    def save(self, 
             path: str | Path = './dataset_index', 
             format: str = 'json',
             paths_type : Literal["absolute", "relative"] = "relative") -> None:
        """
        Save the dataset index to a file. Supported formats: json, yaml, txt.
        """

        # TODO save paths according to type. If relative save relative to dataset_root_path, otherwise absolute paths including dataset_root_path

        # Normalize path
        if isinstance(path, str):
            path = Path(path)

        fmt = format.lower()

        # Warn and strip any existing extension
        existing_ext = path.suffix
        if existing_ext:
            ext_clean = existing_ext.lstrip('.').lower()
            print(f"Warning: provided path already has extension '.{ext_clean}'. This will be overridden by format.")
            path = path.with_suffix('')

        # Append the correct extension
        path = path.with_suffix(f".{fmt}")

        # Build a serializable dict
        index = {
            "dataset_root_paths": (
                [str(p) for p in self.dataset_root_paths]
                if isinstance(self.dataset_root_paths, (list, tuple))
                else str(self.dataset_root_paths)
            ),
            "dataset_format_object": [str(object=obj) for obj in self.dataset_format_objects], # TODO this is not really stringable
            "dataset_names": [str(n) for n in self.dataset_names],
            "dataset_paths": [str(p) for p in self.dataset_paths],
        }

        # Write file according to format
        if fmt == 'json':
            with open(path, 'w') as f:
                json.dump(index, f, indent=4)

        elif fmt in ('yaml', 'yml'):
            with open(path, 'w') as f:
                yaml.safe_dump(index, f)

        elif fmt == 'txt':
            with open(path, 'w') as f:
                f.write(str(self))

        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'json', 'yaml', or 'txt'.")

    @classmethod
    def load(cls, 
             index_tree_path: str | Path,
             dataset_root_path : str | Path | list[Path | str]) -> "DatasetsIndexTree":
        """
        Load and reconstruct a DatasetsIndexTree from a saved index file.
        Supports json, yaml; txt is not supported for now.
        """

        # TODO if dataset_root_path is provided, use it to resolve relative paths in the index file

        # Normalize paths to use pathlib
        index_tree_path = Path(index_tree_path)

        if not index_tree_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_tree_path!s}")

        fmt = index_tree_path.suffix.lstrip('.').lower()
        if fmt == 'json':
            with open(index_tree_path, 'r') as f:
                data = json.load(f)

        elif fmt in ('yaml', 'yml'):
            with open(index_tree_path, 'r') as f:
                data = yaml.safe_load(f)

        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'json' or 'yaml'.")

        # TODO this is a prototype
        # Reconstruct format‐objects if possible
        fmt_objs = []
        for repr_str in data.get("dataset_format_object", []):
            # assume ImagesDatasetFormat has a from_string or equivalent
            fmt_objs.append(ImagesDatasetFormat.from_string(repr_str))

        # build the tree
        root_paths = data.get("dataset_root_paths")
        tree = cls(root_paths, fmt_objs)
        tree.dataset_names = data.get("dataset_names", [])
        tree.dataset_paths = data.get("dataset_paths", [])

        # Rebuild DatasetIndex entries (assume DatasetIndex.load exists)
        tree.dataset_indices = [
            DatasetIndex.load(di_dict)
            for di_dict in data.get("dataset_indices", [])
        ]

        return tree

# %% Dataset base classes
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

        # Setup backend loader
        self.image_backend = image_backend

        if image_backend == 'cv2':
            try:
                import cv2
            except ImportError:
                raise ImportError(
                    "OpenCV (cv2) backend requested, but not installed")

            self._load_img_from_file: Callable = partial(cv2.imread,
                                                         flags=cv2.IMREAD_UNCHANGED)

        elif image_backend == 'pil':
            try:
                from PIL import Image
            except ImportError:
                raise ImportError("PIL backend requested, but not installed")

            self._load_img_from_file = Image.open
        else:
            raise ValueError(f"Unsupported image_backend: {image_backend}")

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

        pattern = os.path.join(root_dir,  f"*{image_meta_ext}")
        image_paths = sorted(glob.glob(pattern))
        camera_path = os.path.join(root_dir, camera_meta_filename)

        return cls(image_paths, camera_path, transform, image_backend)

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_indexed_from_path(self):
        # TODO method to load a single (image, labels) pair from disk
        pass

    def load_image(self, image_path: str) -> dict[str, Any]:
        """
        Load image via selected backend, plus metadata and camera params.
        """
        scene_meta = self._load_yaml(image_path)
        img_filename = scene_meta.get('image', None)

        if img_filename is None:
            raise KeyError(f"'image' key not found in {image_path}")
        img_path = os.path.join(os.path.dirname(image_path), img_filename)

        # Image loading (call backend method)
        image = self._load_img_from_file(img_path)
        if image is None:
            raise FileNotFoundError(
                f"Failed to load image {img_path} with backend {self.image_backend}"
            )

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

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def _process_labels(self, raw_labels: dict[str, Any]) -> Any:
        """
        Placeholder for label processing. Override in subclass.
        """
        raise NotImplementedError(
            "Subclasses should implement _process_labels to process raw labels.")


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


# TODO may be removed entirely
class ImagesLabelsDataset(ImagesLabelsDatasetBase):
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
