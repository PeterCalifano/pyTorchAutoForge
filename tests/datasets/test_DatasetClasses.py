import sys
import os
from pyTorchAutoForge.datasets.DatasetClasses import ImagesLabelsContainer, NormalizeDataMatrix, NormalizationType, DatasetLoaderConfig, DatasetIndex, DatasetsIndexTree

import pytest

def _get_dataset_env_root():
    # Set dataset root
    DATASET_ENV_ROOT = os.getenv('DATASETS')
    print(f"DATASET_ENV_ROOT: {DATASET_ENV_ROOT}")

    if DATASET_ENV_ROOT is None:
        raise ValueError("Environment variable 'DATASETS' is not set.")
    
    return DATASET_ENV_ROOT


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

def test_load_sample_dataset():
    pass


if __name__ == "__main__":
    #pytest.main([__file__])
    #test_load_starnav_collections()

