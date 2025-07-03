import sys
import os
from pyTorchAutoForge.datasets.DatasetClasses import ImagesLabelsContainer, NormalizeDataMatrix, NormalizationType, DatasetLoaderConfig

import pytest

def _get_dataset_env_root():
    # Set dataset root
    DATASET_ENV_ROOT = os.getenv('DATASETS')
    print(f"DATASET_ENV_ROOT: {DATASET_ENV_ROOT}")

    if DATASET_ENV_ROOT is None:
        raise ValueError("Environment variable 'DATASETS' is not set.")
    
    return DATASET_ENV_ROOT

def test_ImagesDatasetConfig():
    pass


def test_ImagesLabelsDatasetBase():
    pass


def test_ImagesLabelsCachedDataset():
    pass

if __name__ == "__main__":
    #pytest.main([__file__])





