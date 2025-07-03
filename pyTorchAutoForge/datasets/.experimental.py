
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
from pyTorchAutoForge.datasets.DatasetClasses import ImagesDatasetType


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
        type_name: dataset_format_types = self.__class__.__name__
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
        raise NotImplementedError(
            "get_name() not implemented yet for ImagesDatasetFormat_PointCloud")


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
        raise NotImplementedError(
            "get_name() not implemented yet for ImagesDatasetFormat_Trajectory")

# %% Dataset index classes


@dataclass
class DatasetIndex:
    dataset_root_path: Path | str | None = None
    dataset_format_objects: dataset_format_types | None = None
    dataset_name: str | None = None

    dataset_inputs_paths: list[str | Path] = field(default_factory=list)
    dataset_targets_paths: list[str | Path] = field(default_factory=list)

    # Optional settings
    img_name_folder: str = "images"
    label_name_folder: str = "labels"
    events_name_folder: str = "events"
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
                 dataset_root_path: Path | str | list[Path |
                                                      str] | None = None,
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
            print(
                f"Dataset root path {self.dataset_root_path} does not exist.")

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
            raise TypeError(
                "dataset_format_object must be an instance of ImagesDatasetFormat.")

        if isinstance(self.dataset_format_objects, list):

            # Check if the dataset format object already exists
            if dataset_format_object in self.dataset_format_objects:
                print(
                    f"Dataset format object {dataset_format_object} already exists in the index.")
                return

            self.dataset_format_objects.append(dataset_format_object)

    def save(self,
             path: str | Path = './dataset_index',
             format: str = 'json',
             paths_type: Literal["absolute", "relative"] = "relative") -> None:
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
            print(
                f"Warning: provided path already has extension '.{ext_clean}'. This will be overridden by format.")
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
            # TODO this is not really stringable
            "dataset_format_object": [str(object=obj) for obj in self.dataset_format_objects],
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
            raise ValueError(
                f"Unsupported format '{format}'. Use 'json', 'yaml', or 'txt'.")

    @classmethod
    def load(cls,
             index_tree_path: str | Path,
             dataset_root_path: str | Path | list[Path | str]) -> "DatasetsIndexTree":
        """
        Load and reconstruct a DatasetsIndexTree from a saved index file.
        Supports json, yaml; txt is not supported for now.
        """

        # TODO if dataset_root_path is provided, use it to resolve relative paths in the index file

        # Normalize paths to use pathlib
        index_tree_path = Path(index_tree_path)

        if not index_tree_path.exists():
            raise FileNotFoundError(
                f"Index file not found: {index_tree_path!s}")

        fmt = index_tree_path.suffix.lstrip('.').lower()
        if fmt == 'json':
            with open(index_tree_path, 'r') as f:
                data = json.load(f)

        elif fmt in ('yaml', 'yml'):
            with open(index_tree_path, 'r') as f:
                data = yaml.safe_load(f)

        else:
            raise ValueError(
                f"Unsupported format '{fmt}'. Use 'json' or 'yaml'.")

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


def test_load_starnav_collections():

    # Get the dataset environment root
    DATASET_ENV_ROOT = _get_dataset_env_root()

    DATASET_ROOT = os.path.join(DATASET_ENV_ROOT, 'StarNavDatasets')

    print(f"DATASET_ROOT: {DATASET_ROOT}")
    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(
            f"Dataset root directory does not exist: {DATASET_ROOT}")

    # Create a configuration for the dataset loader
    dataset_names = [
        name for name in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, name))
    ]

    format_types = []

    # Test building of dataset index
    dataset_indices = []
    for ith, (dataset_name, format_type) in enumerate(zip(dataset_names, format_types)):

        # Build a dataset index for each dataset
        dataset_root_path = os.path.join(DATASET_ROOT)

        tmp_index = DatasetIndex(dataset_root_path,
                                 dataset_name=dataset_name,
                                 dataset_format_objects=format_type,
                                 )

        dataset_indices.append(tmp_index)

    print(f"Available datasets: {dataset_names}")
