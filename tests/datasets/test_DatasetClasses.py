import os
import tempfile
from pyTorchAutoForge.datasets.DatasetClasses import ImagesLabelsContainer, NormalizeDataMatrix, NormalizationType, DatasetLoaderConfig, ImagesDatasetConfig, ImagesLabelsDatasetBase, PTAF_Datakey, DatasetPathsContainer, FetchDatasetPaths, SamplesSelectionCriteria

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from torch.utils.data import DataLoader
import pyTorchAutoForge.datasets.DatasetClasses as dataset_classes_module
from pyTorchAutoForge.datasets.LabelsClasses import LabelsContainer

# %% Auxiliary functions
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

# %% Tests for dataset configuration classes
def test_ImagesDatasetConfig():
    dataset_env_root = _get_dataset_env_root()

    # Define dataset paths
    dataset_root = (os.path.join(dataset_env_root, 'UniformlyScatteredPointCloudsDatasets/Moon'),
                    os.path.join(dataset_env_root, 'UniformlyScatteredPointCloudsDatasets/Ceres'))

    dataset_names = "Dataset_UniformAzElPointCloud_ABRAM_Moon_2500_ID0"
    lbl_vector_data_keys = (PTAF_Datakey.CENTRE_OF_FIGURE,
                            PTAF_Datakey.RANGE_TO_COM)
    
    dset_config = ImagesDatasetConfig(dataset_names_list=dataset_names,
                                      datasets_root_folder=dataset_root,
                                lbl_vector_data_keys=lbl_vector_data_keys,)

    assert dset_config.dataset_names_list == (dataset_names,)
    assert dset_config.datasets_root_folder == dataset_root
    assert dset_config.lbl_vector_data_keys == lbl_vector_data_keys

    return dset_config

def test_ImagesDatasetConfig_path_not_existing():
    dataset_env_root = _get_dataset_env_root()

    # Define dataset paths
    dataset_root = (os.path.join(dataset_env_root, 'UniformlyScatteredPointCloudsDatasets/Moon'),
                    os.path.join(dataset_env_root, 'NonExistentPath/Ceres'))  # Non-existent path

    dataset_names = "Dataset_UniformAzElPointCloud_ABRAM_Moon_2500_ID0"
    lbl_vector_data_keys = (PTAF_Datakey.CENTRE_OF_FIGURE,
                            PTAF_Datakey.RANGE_TO_COM)
    
    with pytest.raises(FileNotFoundError, match="Dataset root folder.*does not exist"):
        dset_config = ImagesDatasetConfig(dataset_names_list=dataset_names,
                                      datasets_root_folder=dataset_root,
                                lbl_vector_data_keys=lbl_vector_data_keys,)


def test_DatasetLoaderConfig_init_with_strings():
    """Test initialization of DatasetLoaderConfig with string values"""
    with patch('os.path.exists', return_value=True):
        config = DatasetLoaderConfig(
            dataset_names_list="dataset1",
            datasets_root_folder="/fake/path",
            lbl_vector_data_keys=(PTAF_Datakey.CENTRE_OF_FIGURE,)
        )

        assert config.dataset_names_list == ("dataset1",)
        assert config.datasets_root_folder == ("/fake/path",)
        assert config.lbl_vector_data_keys == (PTAF_Datakey.CENTRE_OF_FIGURE,)
        assert config.labels_folder_name == "labels"
        assert config.lbl_dtype == torch.float32
        assert config.samples_limit_per_dataset == -1


def test_DatasetLoaderConfig_init_with_lists_and_tuples():
    """Test initialization with lists and tuples"""
    with patch('os.path.exists', return_value=True):
        config = DatasetLoaderConfig(
            dataset_names_list=["dataset1", "dataset2"],
            datasets_root_folder=("/fake/path1", "/fake/path2"),
            lbl_vector_data_keys=(
                PTAF_Datakey.CENTRE_OF_FIGURE, PTAF_Datakey.RANGE_TO_COM)
        )

        assert isinstance(config.dataset_names_list, list)
        assert len(config.dataset_names_list) == 2
        assert isinstance(config.datasets_root_folder, tuple)
        assert len(config.datasets_root_folder) == 2


def test_DatasetLoaderConfig_init_with_paths():
    """Test initialization with Path objects"""
    with patch('os.path.exists', return_value=True):
        path1 = Path("/fake/path1")
        path2 = Path("/fake/path2")

        config = DatasetLoaderConfig(
            dataset_names_list=path1,
            datasets_root_folder=path2,
            lbl_vector_data_keys=(PTAF_Datakey.CENTRE_OF_FIGURE,)
        )

        assert config.dataset_names_list == (path1,)
        assert config.datasets_root_folder == (path2,)


def test_DatasetLoaderConfig_nonexistent_folder():
    """Test validation of dataset root folders existence"""
    with patch('os.path.exists', side_effect=lambda path: path != "/nonexistent/path"):
        with pytest.raises(FileNotFoundError, match="Dataset root folder.*does not exist"):
            DatasetLoaderConfig(
                dataset_names_list="dataset1",
                datasets_root_folder="/nonexistent/path",
                lbl_vector_data_keys=(PTAF_Datakey.CENTRE_OF_FIGURE,)
            )


def test_DatasetLoaderConfig_string_datakeys():
    """Test handling of string data keys by converting to PTAF_Datakey"""
    with patch('os.path.exists', return_value=True):
        config = DatasetLoaderConfig(
            dataset_names_list="dataset1",
            datasets_root_folder="/fake/path",
            lbl_vector_data_keys=("CENTRE_OF_FIGURE", "RANGE_TO_COM")
        )

    # Check all lbl_vector_data_keys are PTAF_Datakey instances
    assert all(isinstance(key, PTAF_Datakey) for key in config.lbl_vector_data_keys)



def test_DatasetLoaderConfig_invalid_string_datakey():
    """Test error handling with invalid string data keys"""
    with patch('os.path.exists', return_value=True):
        with patch.object(PTAF_Datakey, '__getitem__', side_effect=KeyError("INVALID_KEY")):
            with pytest.raises(ValueError, match="Invalid label data key string"):
                DatasetLoaderConfig(
                    dataset_names_list="dataset1",
                    datasets_root_folder="/fake/path",
                    lbl_vector_data_keys=("INVALID_KEY",)
                )


def test_DatasetLoaderConfig_invalid_datakey_type():
    """Test error handling with invalid data key types"""
    with patch('os.path.exists', return_value=True):
        with pytest.raises(TypeError, match="lbl_vector_data_keys must be of type PTAF_Datakey or str"):
            DatasetLoaderConfig(
                dataset_names_list="dataset1",
                datasets_root_folder="/fake/path",
                lbl_vector_data_keys=(123,)  # Not a PTAF_Datakey or str
            )


def test_ImagesDatasetConfig_inheritance():
    """Test that ImagesDatasetConfig inherits from DatasetLoaderConfig"""
    with patch('os.path.exists', return_value=True):
        config = ImagesDatasetConfig(
            dataset_names_list="dataset1",
            datasets_root_folder="/fake/path",
            lbl_vector_data_keys=(PTAF_Datakey.CENTRE_OF_FIGURE,),
            image_dtype=np.uint8,
        )

        # Test inherited attributes
        assert config.dataset_names_list == ("dataset1",)
        assert config.datasets_root_folder == ("/fake/path",)

        # Test class-specific attributes
        assert config.binary_masks_folder_name == "binary_masks"
        assert config.images_folder_name == "images"
        assert config.image_format == "png"
        assert config.image_dtype == np.uint8
        assert config.image_backend == "cv2"
        assert config.convert_rgb_to_grayscale is False
        assert config.intensity_scaling_mode == "dtype"
        assert config.intensity_scale_value is None


def test_ImagesDatasetConfig_intensity_scaling_validation():
    """Test validation of intensity scaling options"""
    with patch('os.path.exists', return_value=True):
        # Valid configurations
        config1 = ImagesDatasetConfig(
            dataset_names_list="dataset1",
            datasets_root_folder="/fake/path",
            lbl_vector_data_keys=(PTAF_Datakey.CENTRE_OF_FIGURE,),
            intensity_scaling_mode="none"
        )
        assert config1.intensity_scaling_mode == "none"

        config2 = ImagesDatasetConfig(
            dataset_names_list="dataset1",
            datasets_root_folder="/fake/path",
            lbl_vector_data_keys=(PTAF_Datakey.CENTRE_OF_FIGURE,),
            intensity_scaling_mode="custom",
            intensity_scale_value=0.5
        )
        assert config2.intensity_scaling_mode == "custom"
        assert config2.intensity_scale_value == 0.5

        # Invalid scaling mode
        with pytest.raises(ValueError, match="Unsupported intensity scaling mode"):
            ImagesDatasetConfig(
                dataset_names_list="dataset1",
                datasets_root_folder="/fake/path",
                lbl_vector_data_keys=(PTAF_Datakey.CENTRE_OF_FIGURE,),
                intensity_scaling_mode="invalid_mode"
            )

        # Custom mode without scale value should still pass (not enforced)
        config3 = ImagesDatasetConfig(
            dataset_names_list="dataset1",
            datasets_root_folder="/fake/path",
            lbl_vector_data_keys=(PTAF_Datakey.CENTRE_OF_FIGURE,),
            intensity_scaling_mode="custom",
            intensity_scale_value=None
        )
        assert config3.intensity_scaling_mode == "custom"

        # Non-custom mode with scale value
        with pytest.raises(ValueError, match="intensity_scale_value must be None"):
            ImagesDatasetConfig(
                dataset_names_list="dataset1",
                datasets_root_folder="/fake/path",
                lbl_vector_data_keys=(PTAF_Datakey.CENTRE_OF_FIGURE,),
                intensity_scaling_mode="dtype",
                intensity_scale_value=0.5)


def _create_test_dataset(root_path: Path,
                         dataset_name: str,
                         stems: list[str],
                         apparent_sizes: dict[str, float] | None = None,
                         bbox_by_stem: dict[str, tuple[float, float, float, float]] | None = None) -> tuple[Path, Path]:
    dataset_path = root_path / dataset_name
    images_path = dataset_path / "images"
    labels_path = dataset_path / "labels"
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        (images_path / f"{stem}.png").write_bytes(b"")
        label_container = LabelsContainer()
        label_container.geometric.bbox_coords_order = "xywh"
        label_container.geometric.bound_box_coordinates = (
            bbox_by_stem.get(stem, (0.0, 0.0, 10.0, 10.0))
            if bbox_by_stem is not None else (0.0, 0.0, 10.0, 10.0)
        )
        label_container.geometric.obj_apparent_size_in_pix = (
            apparent_sizes.get(stem, 0.0) if apparent_sizes is not None else 0.0
        )
        label_container.save_to_yaml(str(labels_path / f"{stem}.yml"))

    return images_path, labels_path


def test_FetchDatasetPaths_basic_collects_all_pairs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        dataset_name = "dataset_basic"
        stems = ["000001", "000002", "000003"]
        _create_test_dataset(root_path, dataset_name, stems)

        result = FetchDatasetPaths(
            dataset_name=dataset_name,
            datasets_root_folder=(str(root_path),),
        )

        assert result.total_num_entries == 3
        assert len(result.img_filepaths) == 3
        assert len(result.lbl_filepaths) == 3
        assert [Path(path).stem for path in result.img_filepaths] == stems
        assert [Path(path).stem for path in result.lbl_filepaths] == stems


def test_FetchDatasetPaths_samples_limit_applies_to_single_dataset():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        dataset_name = "dataset_limit"
        stems = ["000001", "000002", "000003", "000004"]
        _create_test_dataset(root_path, dataset_name, stems)

        result = FetchDatasetPaths(
            dataset_name=dataset_name,
            datasets_root_folder=(str(root_path),),
            samples_limit_per_dataset=2,
        )

        assert result.total_num_entries == 2
        assert len(result.img_filepaths) == 2
        assert len(result.lbl_filepaths) == 2
        assert [Path(path).stem for path in result.img_filepaths] == stems[:2]


def test_FetchDatasetPaths_selection_criteria_filters_by_apparent_size():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        dataset_name = "dataset_apparent_size"
        stems = ["000001", "000002", "000003"]
        apparent_sizes = {"000001": 1.0, "000002": 20.0, "000003": 5.0}
        _create_test_dataset(root_path, dataset_name, stems, apparent_sizes=apparent_sizes)

        result = FetchDatasetPaths(
            dataset_name=dataset_name,
            datasets_root_folder=(str(root_path),),
            selection_criteria=SamplesSelectionCriteria(max_apparent_size=10.0),
        )

        assert [Path(path).stem for path in result.img_filepaths] == ["000001", "000003"]
        assert result.total_num_entries == 2


def test_FetchDatasetPaths_selection_criteria_filters_by_bbox():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        dataset_name = "dataset_bbox"
        stems = ["000001", "000002", "000003"]
        bbox_by_stem = {
            "000001": (0.0, 0.0, 2.0, 2.0),
            "000002": (0.0, 0.0, 10.0, 1.0),
            "000003": (0.0, 0.0, 6.0, 8.0),
        }
        _create_test_dataset(root_path, dataset_name, stems, bbox_by_stem=bbox_by_stem)

        result = FetchDatasetPaths(
            dataset_name=dataset_name,
            datasets_root_folder=(str(root_path),),
            selection_criteria=SamplesSelectionCriteria(min_bbox_width_height=(4.0, 4.0)),
        )

        assert [Path(path).stem for path in result.img_filepaths] == ["000002", "000003"]
        assert result.total_num_entries == 2


def test_FetchDatasetPaths_string_root_folder_supported():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        dataset_name = "dataset_root_as_string"
        stems = ["000001", "000002"]
        _create_test_dataset(root_path, dataset_name, stems)

        result = FetchDatasetPaths(
            dataset_name=dataset_name,
            datasets_root_folder=str(root_path),
        )

        assert [Path(path).stem for path in result.img_filepaths] == stems
        assert result.total_num_entries == 2


def test_FetchDatasetPaths_samples_limit_keeps_multiple_datasets():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        dataset_a = "dataset_A"
        dataset_b = "dataset_B"
        stems_a = ["000001", "000002", "000003"]
        stems_b = ["000004", "000005", "000006"]
        _create_test_dataset(root_path, dataset_a, stems_a)
        _create_test_dataset(root_path, dataset_b, stems_b)

        result = FetchDatasetPaths(
            dataset_name=[dataset_a, dataset_b],
            datasets_root_folder=(str(root_path),),
            samples_limit_per_dataset=2,
        )

        assert result.total_num_entries == 4
        assert result.num_of_entries_in_set == [2, 2]
        assert [Path(path).stem for path in result.img_filepaths] == ["000001", "000002", "000004", "000005"]

# %% Tests for dataset classes
def test_ImagesLabelsDatasetBase():
    dset_config = test_ImagesDatasetConfig()

    # Create dataset object
    dataset = ImagesLabelsDatasetBase(dset_cfg=dset_config)

    assert dataset is not None
    assert dataset.dset_cfg == dset_config

    # Test getitem with a valid index
    img, lbl = dataset[0]
    assert img is not None
    assert lbl is not None

    # Check shape of img is (B,C,H,W) and lbl is (B, num_labels)
    assert len(img.shape) == 3 # Image loaded as (C,H,W)
    assert img.shape[0] == 1
    assert lbl.shape[0] == 3 # Determined by datakeys from dset_config. Output is 1D array.

    # Test dataloader definition and batch fetching
    try:
        from torch.utils.data import DataLoader
        
        # Create a dataloader with the dataset
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Get a batch
        batch_imgs, batch_lbls = next(iter(dataloader))
        
        # Check shapes: should be (batch_size, channels, height, width) for images
        # and (batch_size, num_labels) for labels
        assert batch_imgs.dim() == 4
        assert batch_imgs.shape[0] <= 2  # Batch size might be smaller if dataset has only 1 item
        assert batch_lbls.dim() == 2
        assert batch_lbls.shape[0] <= 2
        assert batch_lbls.shape[1] == 3  # Should match the number of label components
        
    except Exception as e:
        pytest.fail(f"DataLoader test failed with exception: {str(e)}")


def test_ImagesLabelsCachedDataset():
    import numpy as np
    import torch
    from pyTorchAutoForge.datasets.DatasetClasses import ImagesLabelsContainer, ImagesLabelsCachedDataset
    from torchvision.transforms import Compose, ToTensor
    from unittest.mock import patch, MagicMock
    
    # Test with numpy arrays
    images_np = np.random.randint(0, 255, (10, 32, 32), dtype=np.uint8)
    labels_np = np.random.rand(10, 5).astype(np.float32)
    
    container_np = ImagesLabelsContainer(images=images_np, labels=labels_np)
    dataset_np = ImagesLabelsCachedDataset(images_labels=container_np)
    
    assert dataset_np is not None
    assert len(dataset_np) == 10
    
    # Test with torch tensors
    images_torch = torch.randint(0, 255, (10, 32, 32), dtype=torch.uint8)
    labels_torch = torch.rand(10, 5)
    
    container_torch = ImagesLabelsContainer(images=images_torch, labels=labels_torch)
    dataset_torch = ImagesLabelsCachedDataset(images_labels=container_torch)
    
    assert dataset_torch is not None
    assert len(dataset_torch) == 10
    
    # Test with transforms
    simple_transform = Compose([ToTensor()])
    dataset_with_transforms = ImagesLabelsCachedDataset(
        images_labels=container_np, 
        transforms=simple_transform
    )
    
    assert dataset_with_transforms is not None
    
    # Test getitem with normalization
    img, lbl = dataset_torch[0]
    assert img.dtype == torch.float32  # Should be normalized to float
    assert torch.max(img) <= 1.0  # Should be normalized to [0,1]
    
    # Test invalid inputs
    with pytest.raises(TypeError):
        ImagesLabelsCachedDataset(images_labels="not_a_container")
    
    with pytest.raises(ValueError):
        ImagesLabelsCachedDataset(images_labels=None)
    
    # Test not-implemented loading from paths
    with pytest.raises(NotImplementedError):
        ImagesLabelsCachedDataset(images_path="dummy_path", labels_path="dummy_path")
    
    # Test with mismatched batch dimensions
    images_mismatch = torch.randint(0, 255, (10, 32, 32), dtype=torch.uint8)
    labels_mismatch = torch.rand(8, 5)  # Different batch size
    
    with pytest.raises(ValueError):
        container_mismatch = ImagesLabelsContainer(images=images_mismatch, labels=labels_mismatch)
        dataset_mismatch = ImagesLabelsCachedDataset(images_labels=container_mismatch)

    # Test with 3D images (should be unsqueezed to 4D)
    images_3d = torch.randint(0, 255, (10, 32, 32), dtype=torch.uint8)
    labels_3d = torch.rand(10, 5)
    
    container_3d = ImagesLabelsContainer(images=images_3d, labels=labels_3d)
    dataset_3d = ImagesLabelsCachedDataset(images_labels=container_3d)
    
    # The 3D images should have been unsqueezed to 4D [B, C, H, W]
    assert dataset_3d.tensors[0].dim() == 4
    assert dataset_3d.tensors[0].shape[1] == 1  # Channel dimension should be 1


def _build_minimal_paths_container(image_paths: list[str], label_paths: list[str]) -> DatasetPathsContainer:
    return DatasetPathsContainer(
        img_filepaths=image_paths,
        lbl_filepaths=label_paths,
        num_of_entries_in_set=[len(image_paths)],
        total_num_entries=len(image_paths),
    )


def _build_mocked_dataset(monkeypatch: pytest.MonkeyPatch,
                          image_paths: list[str],
                          convert_rgb_to_grayscale: bool,
                          image_backend: str = "cv2",
                          intensity_scaling_mode: str = "none",
                          ) -> ImagesLabelsDatasetBase:
    with patch('os.path.exists', return_value=True):
        dset_cfg = ImagesDatasetConfig(
            dataset_names_list="dummy_set",
            datasets_root_folder="/dummy_root",
            lbl_vector_data_keys=(PTAF_Datakey.PHASE_ANGLE,),
            image_backend=image_backend,  # type:ignore[arg-type]
            intensity_scaling_mode=intensity_scaling_mode,  # type:ignore[arg-type]
            convert_rgb_to_grayscale=convert_rgb_to_grayscale,
        )

    container = _build_minimal_paths_container(
        image_paths=image_paths,
        label_paths=["dummy_lbl.yml"] * len(image_paths),
    )
    monkeypatch.setattr(dataset_classes_module, "FetchDatasetPaths",
                        lambda *args, **kwargs: container)
    monkeypatch.setattr(
        dataset_classes_module.LabelsContainer,
        "load_from_yaml",
        lambda _path: LabelsContainer(),
    )
    return ImagesLabelsDatasetBase(dset_cfg=dset_cfg)


@pytest.mark.skipif(not dataset_classes_module.hasOpenCV, reason="OpenCV is required for cv2 backend tests.")
def test_ImagesLabelsDatasetBase_mixed_rgb_and_grayscale_to_grayscale(monkeypatch: pytest.MonkeyPatch):
    dataset = _build_mocked_dataset(
        monkeypatch=monkeypatch,
        image_paths=["gray.png", "rgb.png"],
        convert_rgb_to_grayscale=True,
        image_backend="cv2",
    )

    image_map = {
        "gray.png": np.random.randint(0, 255, (12, 12), dtype=np.uint8),
        "rgb.png": np.random.randint(0, 255, (12, 12, 3), dtype=np.uint8),
    }
    dataset._load_img_from_file = lambda path: image_map[path]

    img0, _ = dataset[0]
    img1, _ = dataset[1]
    assert img0.shape == (1, 12, 12)
    assert img1.shape == (1, 12, 12)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_img, batch_lbl = next(iter(dataloader))
    assert batch_img.shape == (2, 1, 12, 12)
    assert batch_lbl.shape == (2, 1)


@pytest.mark.skipif(not dataset_classes_module.hasOpenCV, reason="OpenCV is required for cv2 backend tests.")
def test_ImagesLabelsDatasetBase_rgba_to_grayscale(monkeypatch: pytest.MonkeyPatch):
    dataset = _build_mocked_dataset(
        monkeypatch=monkeypatch,
        image_paths=["rgba.png"],
        convert_rgb_to_grayscale=True,
        image_backend="cv2",
    )
    dataset._load_img_from_file = lambda _path: np.random.randint(
        0, 255, (10, 10, 4), dtype=np.uint8
    )

    img, _ = dataset[0]
    assert img.shape == (1, 10, 10)


@pytest.mark.skipif(not dataset_classes_module.hasOpenCV, reason="OpenCV is required for cv2 backend tests.")
def test_ImagesLabelsDatasetBase_unsupported_channels_raise(monkeypatch: pytest.MonkeyPatch):
    dataset = _build_mocked_dataset(
        monkeypatch=monkeypatch,
        image_paths=["bad_channels.png"],
        convert_rgb_to_grayscale=True,
        image_backend="cv2",
    )
    dataset._load_img_from_file = lambda _path: np.random.randint(
        0, 255, (10, 10, 5), dtype=np.uint8
    )

    with pytest.raises(ValueError, match="Unsupported image shape for grayscale conversion"):
        _ = dataset[0]


@pytest.mark.skipif(not dataset_classes_module.hasOpenCV, reason="OpenCV is required for cv2 backend tests.")
def test_ImagesLabelsDatasetBase_disabled_conversion_keeps_rgb(monkeypatch: pytest.MonkeyPatch):
    dataset = _build_mocked_dataset(
        monkeypatch=monkeypatch,
        image_paths=["rgb.png"],
        convert_rgb_to_grayscale=False,
        image_backend="cv2",
    )
    dataset._load_img_from_file = lambda _path: np.random.randint(
        0, 255, (16, 16, 3), dtype=np.uint8
    )

    img, _ = dataset[0]
    assert img.shape == (3, 16, 16)


@pytest.mark.skipif(not dataset_classes_module.hasOpenCV, reason="OpenCV is required for multiprocessing cv2 test.")
def test_ImagesLabelsDatasetBase_worker_collate_with_mixed_inputs():
    import cv2

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_name = "dataset_mixed_inputs"
        images_dir = Path(tmp_dir) / dataset_name / "images"
        labels_dir = Path(tmp_dir) / dataset_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        gray_img = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
        rgb_img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "000001.png"), gray_img)
        cv2.imwrite(str(images_dir / "000002.png"), rgb_img)

        LabelsContainer().save_to_yaml(str(labels_dir / "000001.yml"))
        LabelsContainer().save_to_yaml(str(labels_dir / "000002.yml"))

        dset_cfg = ImagesDatasetConfig(
            dataset_names_list=dataset_name,
            datasets_root_folder=(tmp_dir,),
            lbl_vector_data_keys=(PTAF_Datakey.PHASE_ANGLE,),
            image_backend="cv2",
            intensity_scaling_mode="none",
            convert_rgb_to_grayscale=True,
        )
        dataset = ImagesLabelsDatasetBase(dset_cfg=dset_cfg)
        dataloader = DataLoader(dataset, batch_size=2,
                                shuffle=False, num_workers=2)

        try:
            batch_img, batch_lbl = next(iter(dataloader))
        except PermissionError as exc:
            pytest.skip(
                f"Multiprocessing semaphores are not available in this execution environment: {exc}"
            )
        assert batch_img.shape == (2, 1, 20, 20)
        assert batch_lbl.shape == (2, 1)
            

# %% MANUAL CALLS for debugging

if __name__ == "__main__":
    #pytest.main([__file__])
    ##test_ImagesDatasetConfig()
    #test_ImagesLabelsDatasetBase()
    pass
