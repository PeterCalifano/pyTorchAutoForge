import kornia.augmentation as K
from logging import warning
from warnings import warn
from typing import Any
from collections.abc import Sequence

try:
    import kornia
    from kornia.augmentation import AugmentationSequential
    import kornia.augmentation as K
    import kornia.geometry as KG
    from kornia import augmentation as kornia_aug
    from kornia.constants import DataKey
    from kornia.augmentation import IntensityAugmentationBase2D, GeometricAugmentationBase2D

    has_kornia = True

except ImportError:
    has_kornia = False

from typing import Literal, TypeAlias
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import colorama
from torchvision import transforms
from pyTorchAutoForge.utils.conversion_utils import torch_to_numpy, numpy_to_torch
from pyTorchAutoForge.datasets.AugmentationsBaseClasses import AugsBaseClass

from pyTorchAutoForge.datasets.noise_models.IntensityAugs import RandomSoftBinarizeImage, RandomConvertTextureToShadingLaw, RandomNoiseTexturePattern
from pyTorchAutoForge.datasets.noise_models.NoiseErrorsAugs import RandomGaussianNoiseVariableSigma, PoissonShotNoise

from pyTorchAutoForge.datasets.noise_models.GeometricAugs import BorderAwareRandomAffine


# %% Type aliases
ndArrayOrTensor: TypeAlias = np.ndarray | torch.Tensor
GeometricAugmentationKey: TypeAlias = tuple[type[GeometricAugmentationBase2D], int]

# %% Custom augmentation modules
# TODO modify to be usable by AugmentationSequential? Inherint from _AugmentationBase. Search how to define custom augmentations in Kornia

# DEVNOTE issue with random apply: 0 is not allowed, but 1 is not either because it implies the user MUST specify at LEAST 2 augs. Easy workaround: automatically add a dummy that acts as a placeholder to make it work with 1


class PlaceholderAugmentation(IntensityAugmentationBase2D):

    def __init__(self):
        super().__init__()

    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict[str, torch.Tensor],
                        flags: dict[str, Any],
                        transform: torch.Tensor | None = None) -> torch.Tensor:

        return input

# %% Augmentation helper configuration dataclass


@dataclass
class AugmentationConfig:
    # Input specification
    input_data_keys: list[DataKey]
    keepdim: bool = True
    same_on_batch: bool = False
    random_apply_photometric: bool = False
    random_apply_minmax: tuple[int, int] = (1, -1)
    device: str | None = None  # Device to run augmentations on, if None, uses torch default
    enable_cache_transforms: bool = False

    # Affine roto-translation augmentation (border aware)
    affine_align_corners: bool = False
    affine_fill_value: int = 0  # Fill value for empty pixels after rotation

    rotation_angle: float | tuple[float, float] = (0.0, 360.0)
    rotation_aug_prob: float = 0.0

    # Translation parameters (in pixels)
    shift_aug_prob: float = 0.0
    max_shift_img_fraction: float | tuple[float, float] = (0.5, 0.5)
    translate_distribution_type: Literal["uniform", "normal"] = "uniform"

    # Flip augmentation probability
    hflip_prob: float = 0.0
    vflip_prob: float = 0.0

    # Optional flag to specify if image is already in the torch layout (overrides guess)
    is_torch_layout: bool | None = None

    # Poisson shot noise
    poisson_shot_noise_aug_prob: float = 0.0

    # Gaussian noise
    sigma_gaussian_noise_dn: int | float | tuple[int |
                                                 float, int | float] = 1.0
    gaussian_noise_aug_prob: float = 0.0

    # Gaussian blur
    kernel_size: tuple[int, int] = (5, 5)
    sigma_gaussian_blur: tuple[float, float] = (0.1, 2.0)
    gaussian_blur_aug_prob: float = 0.0

    # Brightness & contrast
    min_max_brightness_factor: tuple[float, float] = (0.8, 1.2)
    min_max_contrast_factor: tuple[float, float] = (0.8, 1.2)
    brightness_aug_prob: float = 0.0
    contrast_aug_prob: float = 0.0

    # Random binarization
    softbinarize_aug_prob: float = 0.0
    softbinarize_thr_quantile: float = 0.01
    softbinarize_blending_factor_minmax: None | tuple[float, float] = (
        0.3, 0.7)

    # Scaling factors for labels
    label_scaling_factors: ndArrayOrTensor | None = None
    datakey_to_scale: DataKey | None = None

    # Special options
    # If True, input is a tuple of (image, other_data)
    # input_is_tuple: bool = False

    # Whether image is normalized (0–1) or raw (0–255)
    is_normalized: bool = True
    # Optional scaling factor. If None, inference attempt based on dtype
    input_normalization_factor: float | None = None
    # Automatic normalization based on input dtype (for images). No scaling for floating point arrays.
    enable_auto_input_normalization: bool = False
    # Input validation module settings
    enable_batch_validation_check: bool = False
    invalid_sample_remedy_action: Literal["discard",
                                          "resample", "original"] = "original"
    # Max attempts to resample invalid images
    max_invalid_resample_attempts: int = 10
    min_num_bright_pixels: int = 500

    def __post_init__(self):

        if self.label_scaling_factors is not None and self.is_torch_layout:
            self.label_scaling_factors = numpy_to_torch(
                self.label_scaling_factors)

        if self.label_scaling_factors is not None and self.datakey_to_scale is None:
            raise ValueError(
                "If label_scaling_factors is provided, datakey_to_scale must also be specified to indicate which entry in the input must be scaled!.")

        # Ensure interpolation mode is enum type
        # if isinstance(self.rotation_interp_mode, str) and \
        #        self.rotation_interp_mode.upper() in ['BILINEAR', 'NEAREST']:
        #    self.rotation_interp_mode = transforms.InterpolationMode[self.rotation_interp_mode.upper(
        #    )]

        # Ensure either translation or flip is enabled. If both, override translation and select flip
        # if self.shift_aug_prob > 0 and (self.hflip_prob > 0 or self.vflip_prob > 0):
        #    print(f"{colorama.Fore.LIGHTYELLOW_EX}WARNING: Both translation and flip augmentations are enabled. Disabling translation and using flip only.{colorama.Style.RESET_ALL}")
        #    self.shift_aug_prob = 0.0

        # Check augs_datakey are all kornia.DataKey
        if not all(isinstance(key, DataKey) for key in self.input_data_keys):
            raise ValueError(
                "All input_data_keys must be instances of DataKey.")

        # Check max_shift_img_fraction is in (0,1)
        if isinstance(self.max_shift_img_fraction, (tuple, list)):
            if not (0.0 <= self.max_shift_img_fraction[0] <= 1.0 and
                    0.0 <= self.max_shift_img_fraction[1] <= 1.0):
                raise ValueError(
                    "max_shift_img_fraction values must be in the range (0, 1) as fraction of the input image size.")
        else:
            if not (0.0 <= self.max_shift_img_fraction <= 1.0):
                raise ValueError(
                    "max_shift_img_fraction values must be in the range (0, 1) as fraction of the input image size.")


@dataclass
class GeometricTransformMetadata:
    """Stores geometric transform metadata computed for one augmented batch."""

    combined_matrix_3x3: torch.Tensor
    per_op_matrices_3x3: dict[GeometricAugmentationKey, torch.Tensor]
    geometric_ops_order: tuple[GeometricAugmentationKey, ...]
    batch_size: int
    device: str


# %% Augmentation helper class
# TODO (PC) add capability to support custom augmentation module by appending it in the user-specified location ("append_custom_module_after = (module, <literal>)" that maps to a specified entry in the augs_ops list. The given module is then inserted into the list at the specified position)
class ImageAugmentationsHelper(nn.Module):
    def __init__(self, augs_cfg: AugmentationConfig):
        super().__init__()

        self.augs_cfg = augs_cfg
        self.device = torch.device(
            augs_cfg.device) if augs_cfg.device is not None else None
        self.num_aug_ops = 0
        self.enable_cache_transforms = augs_cfg.enable_cache_transforms

        # Cache variables
        self._geometric_aug_modules: list[tuple[GeometricAugmentationKey,
                                                GeometricAugmentationBase2D]] = []
        self._last_batch_transform_metadata: GeometricTransformMetadata | None = None

        # TODO add input_data_keys to AugmentationConfig
        # ImageSequential seems not importable from kornia

        # Define kornia augmentation pipeline
        augs_ops: list[GeometricAugmentationBase2D |
                       IntensityAugmentationBase2D] = []

        # TODO categorize augmentations
        geometric_augs_ops: list[GeometricAugmentationBase2D] = []
        photometric_augs_ops: list[IntensityAugmentationBase2D] = []
        optics_augs_ops: list[IntensityAugmentationBase2D] = []
        noise_augs_ops: list[IntensityAugmentationBase2D] = []

        torch_vision_ops = nn.ModuleList()

        # GEOMETRIC AUGMENTATIONS
        # Image flip augmentation
        if augs_cfg.hflip_prob > 0:
            augs_ops.append(K.RandomHorizontalFlip(p=augs_cfg.hflip_prob))

        if augs_cfg.vflip_prob > 0:
            augs_ops.append(K.RandomVerticalFlip(p=augs_cfg.vflip_prob))

        # Random affine (roto-translation-scaling)
        if augs_cfg.shift_aug_prob > 0 or augs_cfg.rotation_aug_prob:

            # Define rotation angles
            if augs_cfg.rotation_aug_prob > 0:
                rotation_degrees = augs_cfg.rotation_angle
            else:
                rotation_degrees = 0.0

            # Define translation value
            if augs_cfg.shift_aug_prob > 0:
                translate_shift = augs_cfg.max_shift_img_fraction if isinstance(augs_cfg.max_shift_img_fraction, tuple) else (
                    augs_cfg.max_shift_img_fraction, augs_cfg.max_shift_img_fraction)
            else:
                translate_shift = (0.0, 0.0)

            # Construct RandomAffine
            base_random_affine = K.RandomAffine(degrees=rotation_degrees,
                                                translate=translate_shift,
                                                p=augs_cfg.rotation_aug_prob,
                                                keepdim=True,
                                                align_corners=augs_cfg.affine_align_corners,
                                                same_on_batch=False)
            # Wrap into BorderAwareRandomAffine
            augs_ops.append(BorderAwareRandomAffine(base_random_affine=base_random_affine,
                                                    num_pix_crossing_detect=10,
                                                    intensity_threshold_uint8=2.0)
                            )

        # INTENSITY AUGMENTATIONS
        # Random brightness
        if augs_cfg.brightness_aug_prob > 0:
            # Random brightness scaling
            augs_ops.append(K.RandomBrightness(brightness=augs_cfg.min_max_brightness_factor,
                                               p=augs_cfg.brightness_aug_prob,
                                               keepdim=True,
                                               clip_output=False))

        # Random contrast
        if augs_cfg.contrast_aug_prob > 0:
            # Random contrast scaling
            augs_ops.append(K.RandomContrast(contrast=augs_cfg.min_max_contrast_factor,
                                             p=augs_cfg.contrast_aug_prob,
                                             keepdim=True,
                                             clip_output=False))

        # Random soft binarization
        if augs_cfg.softbinarize_aug_prob > 0:
            augs_ops.append(RandomSoftBinarizeImage(aug_prob=augs_cfg.softbinarize_aug_prob,
                                                    masking_quantile=augs_cfg.softbinarize_thr_quantile,
                                                    blending_factor_minmax=augs_cfg.softbinarize_blending_factor_minmax))

        # OPTICS AUGMENTATIONS
        # Random Gaussian blur
        if augs_cfg.gaussian_blur_aug_prob > 0:
            # Random Gaussian blur
            augs_ops.append(K.RandomGaussianBlur(kernel_size=augs_cfg.kernel_size,
                                                 sigma=augs_cfg.sigma_gaussian_blur,
                                                 p=augs_cfg.gaussian_blur_aug_prob,
                                                 keepdim=True))

        # NOISE AUGMENTATIONS
        # Image dependent poisson shot noise
        if augs_cfg.poisson_shot_noise_aug_prob > 0:
            # FIXME it seems that possion shot noise cannot be constructed, investigate
            augs_ops.append(PoissonShotNoise(
                probability=augs_cfg.poisson_shot_noise_aug_prob))

        # Random Gaussian white noise (variable sigma)
        if augs_cfg.gaussian_noise_aug_prob > 0:
            # Random Gaussian noise
            augs_ops.append(RandomGaussianNoiseVariableSigma(sigma_noise=augs_cfg.sigma_gaussian_noise_dn,
                                                             gaussian_noise_aug_prob=augs_cfg.gaussian_noise_aug_prob,
                                                             keep_scalar_sigma_fixed=False,
                                                             enable_img_validation_mode=augs_cfg.enable_batch_validation_check,
                                                             validation_min_num_bright_pixels=50,
                                                             validation_pixel_threshold=5.0
                                                             ))

        ####################################################################
        # Add placeholder if needed to prevent errors due to scalar augmentation count
        if len(augs_ops) == 0:
            print(f"{colorama.Fore.LIGHTYELLOW_EX}WARNING: No augmentations defined in augs_ops! Forward pass will not do anything if called.{colorama.Style.RESET_ALL}")

        elif len(augs_ops) == 1:
            # If len of augs_ops == 1 add placeholder augs for random_apply
            augs_ops.append(PlaceholderAugmentation())

        self.num_aug_ops = len(augs_ops)

        # Build AugmentationSequential from nn.ModuleList
        # Use maximum number of augmentations if upper bound not provided
        if augs_cfg.random_apply_photometric:
            if augs_cfg.random_apply_minmax[1] == -1:
                tmp_random_apply_minmax_ = list(augs_cfg.random_apply_minmax)
                tmp_random_apply_minmax_[1] = len(augs_ops) - 1
            else:
                tmp_random_apply_minmax_ = list(augs_cfg.random_apply_minmax)
                tmp_random_apply_minmax_[1] = tmp_random_apply_minmax_[1] - 1

            random_apply_minmax_: tuple[int, int] | bool = (
                tmp_random_apply_minmax_[0], tmp_random_apply_minmax_[1])
        else:
            random_apply_minmax_ = False

        # Transfer all modules to device if specified
        if self.device is not None:

            for aug_op in augs_ops:
                aug_op.to(self.device)

            for aug_op in torch_vision_ops:
                aug_op.to(self.device)

        self._geometric_aug_modules = [
            ((type(aug_op), idx), aug_op)
            for idx, aug_op in enumerate(augs_ops)
            if isinstance(aug_op, GeometricAugmentationBase2D)
        ]

        self.kornia_augs_module = AugmentationSequential(*augs_ops,
                                                         data_keys=augs_cfg.input_data_keys,
                                                         same_on_batch=False,
                                                         keepdim=False,
                                                         random_apply=random_apply_minmax_
                                                         )
        if self.device is not None:
            self.kornia_augs_module.to(self.device)

        # if augs_cfg.append_custom_module_after_ is not None:
        #    pass

        self.torchvision_augs_module = nn.Sequential(
            *torch_vision_ops) if len(torch_vision_ops) > 0 else None
        if self.device is not None and self.torchvision_augs_module is not None:
            self.torchvision_augs_module.to(self.device)

    def _resolve_device(self, *tensors: torch.Tensor) -> torch.device | None:
        """Helper method to resolve device from input or configuration"""
        if self.device is not None:
            return self.device

        for tensor in tensors:
            if torch.is_tensor(tensor):
                return tensor.device

        return None

    def _move_input_to_device(self, input: Any, device: torch.device | None) -> Any:
        """Helper method to move inputs to device"""
        if device is None:
            return input

        if torch.is_tensor(input):
            return input.to(device)

        if isinstance(input, (list, tuple)):
            return type(input)(
                self._move_input_to_device(item, device) for item in input)

        return input

    @staticmethod
    def _build_identity_transform_matrix(batch_size: int,
                                         device: torch.device,
                                         dtype: torch.dtype) -> torch.Tensor:
        """Build a batch of identity 3x3 homogeneous transforms."""
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)

    def _to_homogeneous_transform_matrix(self,
                                         matrix: torch.Tensor | None,
                                         batch_size: int,
                                         device: torch.device,
                                         dtype: torch.dtype) -> torch.Tensor:
        """Normalize a transform matrix into batched homogeneous [B, 3, 3] format."""
        if matrix is None or not torch.is_tensor(matrix):
            return self._build_identity_transform_matrix(batch_size, device, dtype)

        matrix = matrix.to(device=device, dtype=dtype)

        if matrix.ndim == 2 and matrix.shape == (2, 3):
            matrix = matrix.unsqueeze(0)
        elif matrix.ndim == 2 and matrix.shape == (3, 3):
            matrix = matrix.unsqueeze(0)
        elif matrix.ndim != 3:
            warn(
                f"Unexpected transform matrix rank {matrix.ndim}; expected 2D/3D. Falling back to identity.",
                UserWarning,
            )
            return self._build_identity_transform_matrix(batch_size, device, dtype)

        if matrix.shape[-2:] == (2, 3):
            last_row = torch.tensor([0.0, 0.0, 1.0],
                                    device=device,
                                    dtype=dtype).view(1, 1, 3).repeat(matrix.shape[0], 1, 1)
            matrix = torch.cat((matrix, last_row), dim=1)
        elif matrix.shape[-2:] != (3, 3):
            warn(
                f"Unexpected transform matrix shape {tuple(matrix.shape)}; expected [B,2,3] or [B,3,3]. Falling back to identity.",
                UserWarning,
            )
            return self._build_identity_transform_matrix(batch_size, device, dtype)

        if matrix.shape[0] == batch_size:
            return matrix

        if matrix.shape[0] == 1:
            return matrix.repeat(batch_size, 1, 1)

        warn(
            f"Unexpected transform batch dimension {matrix.shape[0]} (expected {batch_size}). Using first transform for all samples.",
            UserWarning,
        )
        return matrix[:1].repeat(batch_size, 1, 1)

    @staticmethod
    def _extract_module_transform_matrix(module: nn.Module) -> torch.Tensor | None:
        """Try extracting the latest transform matrix from a geometric augmentation module."""
        transform = getattr(module, "transform_matrix", None)
        if torch.is_tensor(transform):
            return transform

        transform = getattr(module, "_transform_matrix", None)
        if torch.is_tensor(transform):
            return transform

        transform = getattr(module, "last_transform_matrix_3x3", None)
        if torch.is_tensor(transform):
            return transform

        return None

    def _build_geometric_transform_metadata(self,
                                            batch_size: int,
                                            device: torch.device,
                                            dtype: torch.dtype) -> GeometricTransformMetadata:
        """Build combined and per-op geometric transform metadata for the current batch."""
        per_op_matrices: dict[GeometricAugmentationKey, torch.Tensor] = {}
        geometric_ops_order: list[GeometricAugmentationKey] = []
        combined = self._build_identity_transform_matrix(
            batch_size, device, dtype)

        for op_key, op_module in self._geometric_aug_modules:
            matrix_raw = self._extract_module_transform_matrix(op_module)
            matrix_h = self._to_homogeneous_transform_matrix(
                matrix_raw, batch_size, device, dtype)
            per_op_matrices[op_key] = matrix_h
            geometric_ops_order.append(op_key)
            combined = matrix_h @ combined

        return GeometricTransformMetadata(
            combined_matrix_3x3=combined,
            per_op_matrices_3x3=per_op_matrices,
            geometric_ops_order=tuple(geometric_ops_order),
            batch_size=batch_size,
            device=str(device),
        )

    # images: ndArrayOrTensor | tuple[ndArrayOrTensor, ...],
    # labels: ndArrayOrTensor

    def forward(self,
                *inputs: ndArrayOrTensor | tuple[ndArrayOrTensor, ...],
                return_transform_metadata: bool = False,
                ) -> tuple[ndArrayOrTensor | tuple[ndArrayOrTensor, ...], ndArrayOrTensor] | tuple[
                    tuple[ndArrayOrTensor |
                          tuple[ndArrayOrTensor, ...], ndArrayOrTensor],
                    GeometricTransformMetadata]:
        """
        images: Tensor[B,H,W,C] or [B,C,H,W], or np.ndarray [...,H,W,C]
        labels: Tensor[B, num_points, 2] or np.ndarray matching batch
        returns: shifted+augmented images & labels, same type as input
        """
        # DEVNOTE scaling and rescaling image may be avoided by modifying intensity-related augmentations instead.
        # TODO add check on size of scale factors. If mismatch wrt labels throw informative error!

        # Find the IMAGE datakey position in self.augs_cfg.input_data_keys
        img_index = self.augs_cfg.input_data_keys.index(DataKey.IMAGE)
        # Index inputs to get image
        images_ = inputs[img_index]

        # Processing batches
        with torch.no_grad():
            self._last_batch_transform_metadata = None

            # Detect type, convert to torch Tensor [B,C,H,W], determine scaling factor
            is_numpy = isinstance(images_, np.ndarray)
            img_tensor, to_numpy, scale_factor = self._preprocess_images(
                images_)

            target_device = self._resolve_device(img_tensor)
            if target_device is not None:
                img_tensor = img_tensor.to(target_device)

            # Undo scaling before adding augs if is_normalized
            if scale_factor is not None and self.augs_cfg.is_normalized:
                img_tensor = img_tensor * scale_factor

            # Apply torchvision augmentation module
            if self.torchvision_augs_module is not None:
                warn(
                    "WARNING: torchvision augmentations module is currently not implemented. Interface will likely be deprecated.",
                    UserWarning,
                )
                # TODO how to do labels update in torchvision?
                # img_tensor = self.torchvision_augs_module(img_tensor)

            # Recompose inputs replacing image
            inputs = list(inputs)
            inputs[img_index] = img_tensor

            # Ensure that inputs are on the correct device
            if target_device is not None:
                inputs = [
                    self._move_input_to_device(input, target_device)
                    for input in inputs
                ]

            ##########
            # Unsqueeze keypoints if input is [B, 2], must be [N, 2]
            # DEVNOTE temporary code that requires extension to be more general
            # TODO find a way to actually get keypoints and other entries without to index those manually

            lbl_to_unsqueeze = inputs[1].dim() == 2
            if lbl_to_unsqueeze:
                # Input is (B,2) --> unsqueeze
                inputs[1] = inputs[1].unsqueeze(1)

            keypoints = inputs[1][..., :2]
            additional_entries = inputs[1][...,
                                           2:] if inputs[1].shape[-1] > 2 else None
            inputs[1] = keypoints
            ##########

            if self.num_aug_ops > 0:
                # %% Apply augmentations module
                aug_inputs = self.kornia_augs_module(*inputs)

                # %% Validate and fix augmentations
                if self.augs_cfg.enable_batch_validation_check:
                    aug_inputs = self._validate_fix_input_img(
                        *aug_inputs, original_inputs=inputs)

            else:
                aug_inputs = inputs

            ##########
            # TODO find a way to actually get keypoints and other entries without to index those manually!
            # Concat additional entries to keypoints entry in aug_inputs
            if additional_entries is not None:
                aug_inputs[1] = torch.cat(
                    [aug_inputs[1], additional_entries], dim=2)

            if lbl_to_unsqueeze:
                aug_inputs[1] = aug_inputs[1].squeeze(1)
            ##########

            # Apply inverse scaling if needed
            if scale_factor is not None and (self.augs_cfg.is_normalized or self.augs_cfg.enable_auto_input_normalization):
                aug_inputs[img_index] = aug_inputs[img_index] / scale_factor

            if aug_inputs[img_index].max() > 10:
                warn(
                    f'\033[93mWARNING: image before clamping to [0,1] has values much greater than 1, that are unlikely to result from augmentations. Check flags: is_normalized: {self.augs_cfg.is_normalized}, enable_auto_input_normalization: {self.augs_cfg.enable_auto_input_normalization}.\033[0m')

            # Apply clamping to [0,1]
            aug_inputs[img_index] = torch.clamp(
                aug_inputs[img_index], 0.0, 1.0)

            if self.augs_cfg.label_scaling_factors is not None and self.augs_cfg.datakey_to_scale is not None:

                lbl_index = self.augs_cfg.input_data_keys.index(
                    self.augs_cfg.datakey_to_scale)

                lbl_tensor = aug_inputs[lbl_index]

                # Apply inverse scaling to labels
                aug_inputs[lbl_index] = lbl_tensor / \
                    self.augs_cfg.label_scaling_factors.to(lbl_tensor.device)

            # Cache geometric transform metadata for the latest batch if geometric augmentations were applied
            if (return_transform_metadata or self.enable_cache_transforms) and torch.is_tensor(aug_inputs[img_index]):

                batch_size = int(aug_inputs[img_index].shape[0])
                self._last_batch_transform_metadata = self._build_geometric_transform_metadata(batch_size=batch_size,
                                                                                               device=aug_inputs[img_index].device,
                                                                                               dtype=aug_inputs[img_index].dtype,
                                                                                               )

            # Convert back to numpy if was ndarray
            if to_numpy is True:
                aug_inputs[img_index] = torch_to_numpy(
                    aug_inputs[img_index].permute(0, 2, 3, 1))
                aug_inputs[lbl_index] = torch_to_numpy(aug_inputs[lbl_index])

        if target_device is not None:
            aug_inputs = type(aug_inputs)(self._move_input_to_device(input, target_device)
                                          for input in aug_inputs)

        # Return augmented inputs and optionally transform metadata
        if return_transform_metadata:
            if self._last_batch_transform_metadata is None and torch.is_tensor(aug_inputs[img_index]):
                # Build if not available
                batch_size = int(aug_inputs[img_index].shape[0])
                self._last_batch_transform_metadata = self._build_geometric_transform_metadata(batch_size=batch_size,
                                                                                               device=aug_inputs[img_index].device,
                                                                                               dtype=aug_inputs[img_index].dtype,
                                                                                               )
            return aug_inputs, self._last_batch_transform_metadata

        return aug_inputs

    def Get_last_batch_transform_metadata(self) -> GeometricTransformMetadata | None:
        """Return transform metadata for the latest forward pass."""
        return self._last_batch_transform_metadata

    def Get_last_batch_transform_matrix_3x3(self) -> torch.Tensor | None:
        """Return only the combined geometric transform matrix for the latest batch."""
        if self._last_batch_transform_metadata is None:
            return None
        return self._last_batch_transform_metadata.combined_matrix_3x3

    def _preprocess_images(self,
                           images: ndArrayOrTensor
                           ) -> tuple[torch.Tensor, bool, float]:
        """
        Preprocess images for augmentation.

        Converts input images (numpy arrays or PyTorch tensors) to a standardized
        PyTorch tensor format [B, C, H, W], applying necessary transformations
        such as layout adjustments, normalization, and dtype conversion.

        Args:
            images (ndArrayOrTensor): Input images as numpy arrays or PyTorch tensors.

        Raises:
            ValueError: If the input image shape is unsupported.
            TypeError: If the input type is neither numpy array nor PyTorch tensor.

        Returns:
            tuple[torch.Tensor, bool]: A tuple containing the processed image tensor
            and a boolean indicating whether the input was originally a numpy array.
        """

        scale_factor = 1.0
        if isinstance(images, np.ndarray):

            imgs_array = images.copy()

            # Determine scale factor
            scale_factor = self._determine_scale_factor(images)

            if imgs_array.ndim < 2 or imgs_array.ndim > 4:
                raise ValueError(
                    "Unsupported image shape. Expected 2D, 3D or 4D array.")

            # If numpy and not specified, assume (B, H, W, C) layout, else use flag
            is_numpy_layout: bool = True if imgs_array.shape[-1] in (
                1, 3) else False
            if self.augs_cfg.is_torch_layout is not None:
                is_numpy_layout = not self.augs_cfg.is_torch_layout

            # Perform convert, unsqueeze and permutation according to flags
            imgs_array = torch.from_numpy(imgs_array.astype(np.float32))

            if imgs_array.ndim == 2 and is_numpy_layout:
                # Single grayscale HxW
                imgs_array = imgs_array.unsqueeze(0)  # Expand to (1,H,W)
                imgs_array = imgs_array.unsqueeze(-1)  # Expand to (1,H,W,1)

            elif imgs_array.ndim == 3:

                if is_numpy_layout and self.augs_cfg.is_torch_layout is None:
                    # Single color or grayscale image (H,W,C)
                    imgs_array = imgs_array.unsqueeze(0)  # Expand to (1,H,W,C)
                    imgs_array = imgs_array.permute(
                        0, 3, 1, 2)  # Permute to (B,C,H,W)

                elif not is_numpy_layout:
                    # Torch layout or multiple batch images
                    if self.augs_cfg.is_torch_layout is None:  # Then multiple images, determined by C
                        imgs_array = imgs_array.unsqueeze(
                            1)  # Expand to (B,1,H,W)
                    else:
                        # Multiple grayscale images (B,H,W)
                        # Expand to (B,H,W,1)
                        imgs_array = imgs_array.unsqueeze(-1)
                        imgs_array = imgs_array.permute(
                            0, 3, 1, 2)  # Permute to (B,C,H,W)

            elif imgs_array.ndim == 4:
                if is_numpy_layout:
                    imgs_array = imgs_array.permute(0, 3, 1, 2)
                # else: If not numpy layout, there is nothing to do

            return imgs_array, True, scale_factor

        elif isinstance(images, torch.Tensor):
            imgs_array = images.to(torch.float32)
            scale_factor = self._determine_scale_factor(images)

            if imgs_array.dim() == 4 and imgs_array.shape[-1] in (1, 3):
                # Detect [B,H,W,C] vs [B,C,H,W]
                # Detected numpy layout, permute
                imgs_array = imgs_array.permute(0, 3, 1, 2)

            elif imgs_array.dim() == 3 and imgs_array.shape[-1] in (1, 3):
                # Detect [H,W,C] vs [C,H,W]
                # Detected numpy layout, permute
                imgs_array = imgs_array.permute(2, 0, 1)

            if imgs_array.dim() == 3 and imgs_array.shape[0] in (1, 3):
                imgs_array = imgs_array.unsqueeze(
                    0)  # Unsqueeze batch dimension

            elif imgs_array.dim() == 3:
                imgs_array = imgs_array.unsqueeze(
                    1)  # Unsqueze channels dimension

            return imgs_array, False, scale_factor

        else:
            raise TypeError(
                f"Unsupported image array type. Expected np.ndarray or torch.Tensor, but found {type(images)}")

    def _determine_scale_factor(self, imgs_array: ndArrayOrTensor) -> float:

        dtype = imgs_array.dtype
        scale_factor = 1.0

        if self.augs_cfg.enable_auto_input_normalization == True and \
                self.augs_cfg.input_normalization_factor is None and \
                self.augs_cfg.is_normalized == False:
            Warning(f"{colorama.Fore.LIGHTRED_EX}WARNING: auto input normalization functionality is enabled but input dtype is {dtype} and no coefficient was provided. Cannot infer image scaling automatically.{colorama.Style.RESET_ALL}")

        if self.augs_cfg.input_normalization_factor is not None:
            scale_factor = float(self.augs_cfg.input_normalization_factor)
        else:
            # Guess based on dtype
            if dtype == torch.uint8 or dtype == np.uint8:
                scale_factor = 255.0
            elif dtype == torch.uint16 or dtype == np.uint16:
                scale_factor = 65535.0
            elif dtype == torch.uint32 or dtype == np.uint32:
                scale_factor = 4294967295.0
            # else: keep 1.0

        return scale_factor

    def _validate_fix_input_img(self,
                                *inputs: ndArrayOrTensor | Sequence[ndArrayOrTensor],
                                original_inputs: ndArrayOrTensor | tuple[ndArrayOrTensor, ...]) -> Sequence[ndArrayOrTensor]:
        """
        Validate input images after augmentation. Attempt fix according to selected remedy action.
        """
        # Determine validity of input images according to is_valid_image_ criteria (default or custom)
        # TODO: allow _is_valid_image to be custom by overloading with user-specified function returning a mask of size (B, N). Use a functor to enforce constraints on the method signature
        # TODO: requires extensive testing!

        inputs = list(inputs)  # FIXME improve this assignment, typing is wrong

        img_index = self.augs_cfg.input_data_keys.index(DataKey.IMAGE)
        lbl_index = self.augs_cfg.input_data_keys.index(
            DataKey.KEYPOINTS)  # TODO (PC) absolutely requires extension!

        # Scan for invalid samples
        is_valid_mask = self._is_valid_image(*inputs)
        invalid_indices = (~is_valid_mask).nonzero(as_tuple=True)[0]

        if not is_valid_mask.all():
            print(f"\r{colorama.Fore.LIGHTYELLOW_EX}WARNING: augmentation validation found {(~is_valid_mask).sum()} invalid samples. Attempting to fix with remedy action: '{self.augs_cfg.invalid_sample_remedy_action}'.{colorama.Style.RESET_ALL}")
        else:
            return inputs

        # If any invalid sample, execute remedy action
        match self.augs_cfg.invalid_sample_remedy_action.lower():

            case "discard":
                # Remove invalid samples from inputs by eliminating invalid indices (reduce batch size!)
                new_inputs = [input[is_valid_mask] for input in inputs]

            case "resample":
                not_all_valid = True

                # Resample until all valid
                iter_counter = 0
                # TODO need to pass in the original of the invalid not the actual invalid!
                invalid_original = [input[~is_valid_mask]
                                    for input in original_inputs]

                while not_all_valid:

                    # Rerun augs module on invalid samples
                    new_aug_inputs_ = self.kornia_augs_module(
                        *invalid_original)

                    # Check new validity mask
                    is_valid_mask_tmp = self._is_valid_image(*new_aug_inputs_)

                    not_all_valid = not (is_valid_mask_tmp.all())

                    if iter_counter == self.augs_cfg.max_invalid_resample_attempts:
                        raise RuntimeError(
                            f"Max invalid resample attempts reached: {iter_counter}. Augmentation helper is not able to provide a fully valid batch. Please verify your augmentation configuration.")

                    if not_all_valid:
                        print(f"{colorama.Fore.LIGHTYELLOW_EX}WARNING: attempt #{iter_counter}/{self.augs_cfg.max_invalid_resample_attempts-1} failed. Current number of invalid samples: {(~is_valid_mask_tmp).sum().float()}.{colorama.Style.RESET_ALL}\n")

                    iter_counter += 1

                # Reallocate new samples in inputs
                for i in range(len(inputs)):
                    inputs[i][invalid_indices] = new_aug_inputs_[i]

                new_inputs = inputs

            case "original":
                # Replace invalid samples with original inputs
                for i, (is_valid, orig_img, orig_lbl) in enumerate(zip(is_valid_mask,
                                                                       original_inputs[img_index],
                                                                       original_inputs[lbl_index])):
                    if not is_valid:
                        inputs[img_index][i] = orig_img
                        inputs[lbl_index][i] = orig_lbl

                new_inputs = inputs

        return list(new_inputs)

    def _is_valid_image(self, *inputs: ndArrayOrTensor | tuple[ndArrayOrTensor, ...]) -> torch.Tensor:
        """
        Check validity of augmented images in a batch.

        This method computes the mean pixel value for each image in the batch and considers
        an image valid if its mean is above a small threshold (default: 1E-3). It returns a
        boolean mask indicating which images are valid, and a tuple of invalid samples
        extracted from the inputs.

        Args:
            *inputs: One or more tensors or tuples of tensors, where the image tensor is
                expected at the index corresponding to DataKey.IMAGE in the input_data_keys.

        Returns:
            is_valid_mask (torch.Tensor): Boolean tensor of shape (B,) indicating validity per image.
        """

        # Compute mean across channels and spatial dims
        img_index = self.augs_cfg.input_data_keys.index(DataKey.IMAGE)

        image_tensor = inputs[img_index]
        if not torch.is_tensor(image_tensor):
            raise TypeError("Expected image input to be a torch.Tensor during validation.")
        B = image_tensor.shape[0]

        # A threshold to detect near-black images (tune if needed)
        # TODO remove hardcoded threshold. It also assumes that the value of intensity is NOT [0,1]!

        is_pixel_bright_count_mask = (
            torch.abs(image_tensor) > 1).view(B, -1).sum(dim=1)
        is_valid_mask = is_pixel_bright_count_mask >= self.augs_cfg.min_num_bright_pixels

        return is_valid_mask

    # Overload "to" method
    def to(self, *args, **kwargs):
        """Overload to method to apply to all submodules."""
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
            self.augs_cfg.device = str(device)

        super().to(*args, **kwargs)
        self.kornia_augs_module.to(*args, **kwargs)

        if self.torchvision_augs_module is not None:
            self.torchvision_augs_module.to(*args, **kwargs)

        return self

    # STATIC METHODS
    @staticmethod
    def Update_clockwise_angle_from_positive_x_using_geometric_transform(
        clockwise_angle_from_positive_x_rad_batch: torch.Tensor,
        geometric_transform_matrix_batch: torch.Tensor,
        wrap_output_to_0_2pi: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Update clockwise angles from +X using the full geometric affine transform.

        The affine translation component is ignored because directions are affected
        only by the linear part of the transform.

        Args:
            clockwise_angle_from_positive_x_rad_batch: Angle tensor with shape
                ``[B]`` or ``[B, 1]`` in radians.
            geometric_transform_matrix_batch: Affine transform matrix with shape
                ``[B, 3, 3]``, ``[B, 2, 3]``, ``[3, 3]`` or ``[2, 3]``.
            wrap_output_to_0_2pi: If True, wrap output angles to ``[0, 2*pi)``.
            eps: Threshold used to detect degenerate transformed direction vectors.

        Returns:
            Updated angle tensor with the same shape as input.
        """

        # Get label shape
        original_angle_tensor_shape = clockwise_angle_from_positive_x_rad_batch.shape
        clockwise_angle_from_positive_x_rad_batch = clockwise_angle_from_positive_x_rad_batch.reshape(
            -1)

        # Extract affine transform data
        if geometric_transform_matrix_batch.ndim == 2:
            geometric_transform_matrix_batch = geometric_transform_matrix_batch.unsqueeze(
                0)

        if geometric_transform_matrix_batch.ndim != 3:
            raise ValueError(
                "geometric_transform_matrix_batch must have shape [B,3,3], [B,2,3], [3,3] or [2,3]."
            )

        if geometric_transform_matrix_batch.shape[-2:] == (3, 3):
            geometric_linear_component_batch_2x2 = geometric_transform_matrix_batch[:, :2, :2]
        elif geometric_transform_matrix_batch.shape[-2:] == (2, 3):
            geometric_linear_component_batch_2x2 = geometric_transform_matrix_batch[:, :2, :2]
        else:
            raise ValueError(
                f"Unsupported transform matrix shape: {tuple(geometric_transform_matrix_batch.shape)}."
            )

        # Compose 2x2 rotation matrix from affine transform
        num_angle_samples = clockwise_angle_from_positive_x_rad_batch.shape[0]
        num_transform_samples = geometric_linear_component_batch_2x2.shape[0]

        if num_transform_samples == 1 and num_angle_samples > 1:
            geometric_linear_component_batch_2x2 = geometric_linear_component_batch_2x2.expand(
                num_angle_samples, -1, -1
            )
        elif num_transform_samples != num_angle_samples:
            raise ValueError(
                f"Mismatch between number of angles ({num_angle_samples}) and transforms ({num_transform_samples})."
            )

        geometric_linear_component_batch_2x2 = geometric_linear_component_batch_2x2.to(
            device=clockwise_angle_from_positive_x_rad_batch.device,
            dtype=clockwise_angle_from_positive_x_rad_batch.dtype,
        )

        # Apply linear part of affine transform to input direction vectors
        input_direction_x_component_batch = torch.cos(
            clockwise_angle_from_positive_x_rad_batch)
        input_direction_y_component_batch = torch.sin(
            clockwise_angle_from_positive_x_rad_batch)

        transformed_direction_x_component_batch = (
            geometric_linear_component_batch_2x2[:, 0,
                                                 0] * input_direction_x_component_batch
            + geometric_linear_component_batch_2x2[:,
                                                   0, 1] * input_direction_y_component_batch
        )
        transformed_direction_y_component_batch = (
            geometric_linear_component_batch_2x2[:, 1,
                                                 0] * input_direction_x_component_batch
            + geometric_linear_component_batch_2x2[:,
                                                   1, 1] * input_direction_y_component_batch
        )

        transformed_direction_squared_norm_batch = (
            transformed_direction_x_component_batch.square()
            + transformed_direction_y_component_batch.square()
        )

        # Check for degenerate transformed directions and replace with input direction if needed to avoid NaNs in angle computation
        is_degenerate_transform_sample_batch = transformed_direction_squared_norm_batch <= eps

        if torch.any(is_degenerate_transform_sample_batch):
            transformed_direction_x_component_batch = torch.where(
                is_degenerate_transform_sample_batch,
                input_direction_x_component_batch,
                transformed_direction_x_component_batch,
            )
            transformed_direction_y_component_batch = torch.where(
                is_degenerate_transform_sample_batch,
                input_direction_y_component_batch,
                transformed_direction_y_component_batch,
            )

        updated_clockwise_angle_from_positive_x_rad_batch = torch.atan2(
            transformed_direction_y_component_batch,
            transformed_direction_x_component_batch,
        )

        # Wrap output angles to [0, 2*pi) if requested
        if wrap_output_to_0_2pi:
            updated_clockwise_angle_from_positive_x_rad_batch = torch.remainder(
                updated_clockwise_angle_from_positive_x_rad_batch,
                2 * torch.pi,
            )

        return updated_clockwise_angle_from_positive_x_rad_batch.reshape(original_angle_tensor_shape)

    @staticmethod
    def Shift_image_point_batch(augs_cfg: AugmentationConfig,
                                images: torch.Tensor,
                                labels: ndArrayOrTensor
                                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
            images: [B,C,H,W]
            labels: torch.Tensor[B,2] or np.ndarray
            returns: shifted images & labels in torch.Tensor
        """

        B, C, H, W = images.shape

        if len(labels.shape) > 2:
            raise NotImplementedError(
                "Current implementation is tailored to translate single point label [Bx2], but got: ", labels.shape)

        # Convert labels to tensor [B,N,2]
        lbl = numpy_to_torch(labels).float() if isinstance(
            labels, np.ndarray) else labels.float()
        assert (
            lbl.shape[0] == B), f"Label batch size {lbl.shape[0]} does not match image batch size {B}."

        # Sample shifts for each batch: dx ∈ [-max_x, max_x], same for dy
        if isinstance(augs_cfg.max_shift_img_fraction, (tuple, list)):
            max_x, max_y = augs_cfg.max_shift_img_fraction
        else:
            max_x, max_y = (augs_cfg.max_shift_img_fraction,
                            augs_cfg.max_shift_img_fraction)

        if augs_cfg.translate_distribution_type == "uniform":
            # Sample shifts by applying 0.99 margin
            dx = torch.randint(-int(max_x), int(max_x)+1, (B,))
            dy = torch.randint(-int(max_y), int(max_y)+1, (B,))

        elif augs_cfg.translate_distribution_type == "normal":
            # Sample shifts from normal distribution
            dx = torch.normal(mean=0.0, std=max_x, size=(B,))
            dy = torch.normal(mean=0.0, std=max_y, size=(B,))
        else:
            raise ValueError(
                f"Unsupported distribution type: {augs_cfg.translate_distribution_type}. Supported types are 'uniform' and 'normal'.")

        shifted_imgs = images.new_zeros(images.shape)

        # TODO improve this method, currently not capable of preventing the object to exit the plane. Also, can be optimized with tensor ops
        for i in range(B):
            ox, oy = int(dx[i]), int(dy[i])

            # Apply saturation to avoid out of bounds
            src_x1 = max(0, -ox)
            src_x2 = min(W, W-ox)
            src_y1 = max(0, -oy)
            src_y2 = min(H, H-oy)

            # Compute destination crop coords
            dst_x1 = max(0, ox)
            dst_x2 = dst_x1 + (src_x2-src_x1)
            dst_y1 = max(0, oy)
            dst_y2 = dst_y1 + (src_y2-src_y1)

            # Copy crop in new image
            shifted_imgs[i, :, dst_y1:dst_y2, dst_x1:dst_x2] = images[i,
                                                                      :, src_y1:src_y2, src_x1:src_x2]

            # Shift points labels
            lbl[i, 0:2] = lbl[i, 0:2] + \
                torch.tensor([ox, oy], dtype=lbl.dtype, device=lbl.device)

        return shifted_imgs, lbl

# %% Prototypes TODO
# Reference for implementation: Gow, 2007, "A Comprehensive tools for modeling CMOS image sensor-noise performance", IEEE Transactions on Electron Devices, Vol. 54, No. 6


class ImageNormalizationCoeff(Enum):
    """Enum for image normalization types."""
    SOURCE = -1.0
    NONE = 1.0
    UINT8 = 255.0
    UINT16 = 65535.0
    UINT32 = 4294967295.0


class ImageNormalization():
    """ImageNormalization class.

    This class normalizes image tensors using the specified normalization type.

    Attributes:
        normalization_type (ImageNormalizationType): The type of normalization to apply to the image.
    """

    def __init__(self, normalization_type: ImageNormalizationCoeff = ImageNormalizationCoeff.NONE):
        self.normalization_type = normalization_type

    def __call__(self, image: torch.Tensor) -> torch.Tensor:

        # Check image datatype, if not float, normalize using dtype
        if image.dtype != torch.float32 and image.dtype != torch.float64:

            # Get datatype and select normalization
            if self.normalization_type == ImageNormalizationCoeff.SOURCE and self.normalization_type is not ImageNormalizationCoeff.NONE:

                if image.dtype == torch.uint8:
                    self.normalization_type = ImageNormalizationCoeff.UINT8

                elif image.dtype == torch.uint16:
                    self.normalization_type = ImageNormalizationCoeff.UINT16

                elif image.dtype == torch.uint32:
                    self.normalization_type = ImageNormalizationCoeff.UINT32
                else:
                    raise ValueError(
                        "Normalization type selected as SOURCE but image type is not uint8, uint16 or uint32. Cannot determine normalization value")

        # Normalize image to range [0,1]
        if self.normalization_type.value < 0.0:
            raise ValueError(
                "Normalization for images value cannot be negative.")

        return image / self.normalization_type.value


############################################################################################################################
# Standalone factory function for AugmentationSequential
def build_kornia_augs(sigma_noise: float, sigma_gaussian_blur: tuple | float = (0.0001, 1.0),
                      brightness_factor: tuple | float = (0.0001, 0.01),
                      contrast_factor: tuple | float = (0.0001, 0.01),
                      use_cfg_factory: bool = True,
                      augs_cfg: AugmentationConfig | None = None) -> nn.Module:
    """Deprecated: prefer AugmentationConfig/ImageAugmentationsHelper; legacy path kept for reference."""

    brightness_min, brightness_max = brightness_factor if isinstance(
        brightness_factor, tuple) else (brightness_factor, brightness_factor)
    contrast_min, contrast_max = contrast_factor if isinstance(
        contrast_factor, tuple) else (contrast_factor, contrast_factor)
    sigma_gaussian_blur_min, sigma_gaussian_blur_max = sigma_gaussian_blur if isinstance(
        sigma_gaussian_blur, tuple) else (sigma_gaussian_blur, sigma_gaussian_blur)

    if use_cfg_factory:
        if augs_cfg is None:
            augs_cfg = AugmentationConfig(
                input_data_keys=[DataKey.IMAGE],
                min_max_brightness_factor=(brightness_min, brightness_max),
                brightness_aug_prob=1.0,
                min_max_contrast_factor=(contrast_min, contrast_max),
                contrast_aug_prob=1.0,
                kernel_size=(5, 5),
                sigma_gaussian_blur=(sigma_gaussian_blur_min,
                                     sigma_gaussian_blur_max),
                gaussian_blur_aug_prob=0.75,
                sigma_gaussian_noise_dn=sigma_noise,
                gaussian_noise_aug_prob=0.75,
            )

        augs_ops: list[GeometricAugmentationBase2D |
                       IntensityAugmentationBase2D] = []

        if augs_cfg.hflip_prob > 0:
            augs_ops.append(K.RandomHorizontalFlip(p=augs_cfg.hflip_prob))

        if augs_cfg.vflip_prob > 0:
            augs_ops.append(K.RandomVerticalFlip(p=augs_cfg.vflip_prob))

        if augs_cfg.shift_aug_prob > 0 or augs_cfg.rotation_aug_prob:

            if augs_cfg.rotation_aug_prob > 0:
                rotation_degrees = augs_cfg.rotation_angle
            else:
                rotation_degrees = 0.0

            if augs_cfg.shift_aug_prob > 0:
                translate_shift = augs_cfg.max_shift_img_fraction if isinstance(augs_cfg.max_shift_img_fraction, tuple) else (
                    augs_cfg.max_shift_img_fraction, augs_cfg.max_shift_img_fraction)
            else:
                translate_shift = (0.0, 0.0)

            base_random_affine = K.RandomAffine(degrees=rotation_degrees,
                                                translate=translate_shift,
                                                p=augs_cfg.rotation_aug_prob,
                                                keepdim=True,
                                                align_corners=augs_cfg.affine_align_corners,
                                                same_on_batch=False)
            augs_ops.append(BorderAwareRandomAffine(base_random_affine=base_random_affine,
                                                    num_pix_crossing_detect=10,
                                                    intensity_threshold_uint8=2.0)
                            )

        if augs_cfg.brightness_aug_prob > 0:
            augs_ops.append(K.RandomBrightness(brightness=augs_cfg.min_max_brightness_factor,
                                               p=augs_cfg.brightness_aug_prob,
                                               keepdim=True,
                                               clip_output=False))

        if augs_cfg.contrast_aug_prob > 0:
            augs_ops.append(K.RandomContrast(contrast=augs_cfg.min_max_contrast_factor,
                                             p=augs_cfg.contrast_aug_prob,
                                             keepdim=True,
                                             clip_output=False))

        if augs_cfg.softbinarize_aug_prob > 0:
            augs_ops.append(RandomSoftBinarizeImage(aug_prob=augs_cfg.softbinarize_aug_prob,
                                                    masking_quantile=augs_cfg.softbinarize_thr_quantile,
                                                    blending_factor_minmax=augs_cfg.softbinarize_blending_factor_minmax))

        if augs_cfg.gaussian_blur_aug_prob > 0:
            augs_ops.append(K.RandomGaussianBlur(kernel_size=augs_cfg.kernel_size,
                                                 sigma=augs_cfg.sigma_gaussian_blur,
                                                 p=augs_cfg.gaussian_blur_aug_prob,
                                                 keepdim=True))

        if augs_cfg.poisson_shot_noise_aug_prob > 0:
            augs_ops.append(PoissonShotNoise(
                probability=augs_cfg.poisson_shot_noise_aug_prob))

        if augs_cfg.gaussian_noise_aug_prob > 0:
            augs_ops.append(RandomGaussianNoiseVariableSigma(sigma_noise=augs_cfg.sigma_gaussian_noise_dn,
                                                             gaussian_noise_aug_prob=augs_cfg.gaussian_noise_aug_prob,
                                                             keep_scalar_sigma_fixed=False,
                                                             enable_img_validation_mode=augs_cfg.enable_batch_validation_check,
                                                             validation_min_num_bright_pixels=50,
                                                             validation_pixel_threshold=5.0
                                                             ))

        if len(augs_ops) == 0:
            print(f"{colorama.Fore.LIGHTYELLOW_EX}WARNING: No augmentations defined in augs_ops! Forward pass will not do anything if called.{colorama.Style.RESET_ALL}")
        elif len(augs_ops) == 1:
            augs_ops.append(PlaceholderAugmentation())

        if augs_cfg.random_apply_photometric:
            if augs_cfg.random_apply_minmax[1] == -1:
                tmp_random_apply_minmax_ = list(augs_cfg.random_apply_minmax)
                tmp_random_apply_minmax_[1] = len(augs_ops) - 1
            else:
                tmp_random_apply_minmax_ = list(augs_cfg.random_apply_minmax)
                tmp_random_apply_minmax_[1] = tmp_random_apply_minmax_[1] - 1

            random_apply_minmax_: tuple[int, int] | bool = (
                tmp_random_apply_minmax_[0], tmp_random_apply_minmax_[1])
        else:
            random_apply_minmax_ = False

        if augs_cfg.device is not None:
            for aug_op in augs_ops:
                aug_op.to(augs_cfg.device)

        return AugmentationSequential(*augs_ops,
                                      data_keys=augs_cfg.input_data_keys,
                                      same_on_batch=False,
                                      keepdim=False,
                                      random_apply=random_apply_minmax_
                                      ).to(augs_cfg.device)

    random_brightness = kornia_aug.RandomBrightness(brightness=(
        brightness_min, brightness_max),
        clip_output=False,
        same_on_batch=False,
        p=1.0,
        keepdim=True)

    random_contrast = kornia_aug.RandomContrast(contrast=(
        contrast_min, contrast_max),
        clip_output=False,
        same_on_batch=False,
        p=1.0,
        keepdim=True)

    gaussian_blur = kornia_aug.RandomGaussianBlur(
        (5, 5), (sigma_gaussian_blur_min, sigma_gaussian_blur_max), p=0.75, keepdim=True)

    gaussian_noise = kornia_aug.RandomGaussianNoise(
        mean=0.0, std=sigma_noise, p=0.75, keepdim=True)

    return torch.nn.Sequential(random_brightness, random_contrast, gaussian_blur, gaussian_noise)


# %% Code for development
if __name__ == "__main__":
    pass
