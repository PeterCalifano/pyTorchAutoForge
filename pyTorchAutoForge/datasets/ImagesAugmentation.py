from logging import warning
from warnings import warn
from typing import Any

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
from torch import nn, Tensor
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import colorama
from torchvision import transforms
from pyTorchAutoForge.utils.conversion_utils import torch_to_numpy, numpy_to_torch
from pyTorchAutoForge.datasets.DataAugmentation import AugsBaseClass

# %% Type aliases
ndArrayOrTensor: TypeAlias = np.ndarray | torch.Tensor

# %% Custom augmentation modules
# TODO modify to be usable by AugmentationSequential? Inherint from _AugmentationBase. Search how to define custom augmentations in Kornia

# DEVNOTE issue with random apply: 0 is not allowed, but 1 is not either because it implies the user MUST specify at LEAST 2 augs. Easy workaround: automatically add a dummy that acts as a placeholder to make it work with 1


class PlaceholderAugmentation(IntensityAugmentationBase2D):

    def __init__(self):
        super().__init__()

    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict,
                        flags: dict,
                        transform: torch.Tensor) -> torch.Tensor:

        return input

# %% Intensity augmentations


class PoissonShotNoise(IntensityAugmentationBase2D):
    """
    Applies Poisson shot noise to a batch of images.

    This module simulates photon shot noise, where the variance of the noise 
    is proportional to the pixel intensity. The noise is applied to a random 
    subset of the batch based on the specified probability.

    Args:
        nn (torch.nn.Module): Base class for all neural network modules.

    Attributes:
        probability (float): Probability of applying Poisson noise to each 
            image in the batch.

    Methods:
        forward(imgs_array: torch.Tensor) -> torch.Tensor:
            Applies Poisson shot noise to the input batch of images.

    Example:
        >>> noise = PoissonShotNoise(probability=0.5)
        >>> noisy_images = noise(images)
    """

    def __init__(self, probability: float = 0.0):
        super().__init__(p=probability)

    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict,
                        flags: dict,
                        transform: torch.Tensor) -> torch.Tensor:
        """
        Applies Poisson shot noise to the input batch of images.

        Args:
            x (torch.Tensor | tuple[torch.Tensor]): Input images as a tensor or tuple of tensors.
            labels (torch.Tensor | tuple[torch.Tensor] | None, optional): Optional labels associated with the images.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: Images with Poisson shot noise applied, or a tuple containing such images.

        """

        # TODO modify to handle tuple inputs

        # Randomly sample a boolean mask to index batch size
        B = input.shape[0]
        device, dtype = input.device, input.dtype

        # Pixel value is the variance of the Photon Shot Noise (higher where brighter).
        # Therefore, the mean rate parameter mu is equal to the DN at the specific pixel.
        photon_shot_noise = torch.poisson(x)

        # Sum noise to the original images according to mask
        x += photon_shot_noise

        return x

    def generate_parameters(self, shape: torch.Size) -> dict:
        return {}  # Not needed for this kind of noise

    def compute_transformation(self, input: torch.Tensor, params: dict, flags: dict) -> torch.Tensor:
        return torch.empty(0)  # Not used in IntensityAugmentationBase2D


class RandomGaussianNoiseVariableSigma(IntensityAugmentationBase2D):
    """
    Applies per-sample Gaussian noise with variable sigma.
    This augmentation adds Gaussian noise to each sample in a batch, where the standard deviation (sigma) can be a scalar, a (min, max) tuple for random sampling, or a per-sample array/tensor. The noise is applied to each sample with a specified probability.

    Args:
        sigma_noise (float or tuple[float, float]): Standard deviation of the Gaussian noise. Can be a scalar
            or a tuple specifying the (min, max) range for random sampling per sample.
        gaussian_noise_aug_prob (float, optional): Probability of applying noise to each sample. Defaults to 0.5.

    Methods:
        forward(x, labels=None):
            Applies Gaussian noise to the input tensor with variable sigma per sample.

    Example:
        >>> aug = RandomGaussianNoiseVariableSigma(sigma_noise=(0.1, 0.5), gaussian_noise_aug_prob=0.7)
        >>> noisy_imgs = aug(images)
    """

    def __init__(self,
                 sigma_noise: float | tuple[float, float],
                 gaussian_noise_aug_prob: float = 0.5):
        super().__init__(p=gaussian_noise_aug_prob)

        self.sigma_gaussian_noise_dn = sigma_noise

    # def forward(self, x: torch.Tensor | tuple[torch.Tensor],
    #            labels: torch.Tensor | tuple[torch.Tensor] | None = None) -> torch.Tensor | tuple[torch.Tensor, ...]:

    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict[str, torch.Tensor],
                        flags: dict[str, Any],
                        transform: torch.Tensor | None) -> torch.Tensor:

        B, C, H, W = input.shape
        device = input.device

        # Determine sigma per sample
        sigma = self.sigma_gaussian_noise_dn

        if isinstance(sigma, tuple):
            min_s, max_s = sigma
            sigma_array = (max_s - min_s) * \
                torch.rand(B, device=device) + min_s
        else:
            sigma_array = torch.full((B,), float(sigma), device=device)

        sigma_array = sigma_array.view(B, 1, 1, 1)
        noise = torch.randn_like(input) * sigma_array

        return input + noise


class RandomNoiseTexturePattern(IntensityAugmentationBase2D):
    """
    Randomly fills the masked region with structured noise. 
    Expects input tensor of shape (B, C+1, H, W):
      - first C channels: image
      - last    1 channel: binary mask (0=keep, 1=replace)
    Returns the same shape.
    """

    def __init__(self, 
                 noise_texture_aug_prob: float = 0.5,
                 randomization_prob : float = 0.6,
                 masking_quantile: float = 0.85) -> None:
        super().__init__(p=noise_texture_aug_prob)

        self.randomization_prob = randomization_prob
        self.masking_quantile = masking_quantile
    
    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict[str, torch.Tensor],
                        flags: dict[str, Any],
                        transform: torch.Tensor | None) -> torch.Tensor:
        
        img_batch = input[0]
        B = input.shape[0]
        device, dtype = img_batch.device, img_batch.dtype

        # Compute masks to select target patches
        brightness = img_batch.mean(dim=1, keepdim=True)  # B×1×H×W
        brightness_thr = (brightness.view(
            B, 1, -1)).quantile(self.masking_quantile, dim=2, keepdim=True)

        img_mask = (brightness > brightness_thr).float()

        # TODO implement noise patterns to apply to image patches

        # Re-attach mask channel for downstream modules
        return out_imgs


# %% Geometric augmentations
# TODO (PC) move translate_batch to this custom augmentation class, add rotation and extend for multiple points labels shift
class RandomImageLabelsRotoTranslation(GeometricAugmentationBase2D):
    """
    RandomImageLabelsRotoTranslation _summary_

    _extended_summary_

    :param AugsBaseClass: _description_
    :type AugsBaseClass: _type_
    """

    def __init__(self,
                 angles: float | tuple[float, float] = (0.0, 360.0),
                 distribution_type: Literal["uniform", "normal"] = "uniform"):
        super().__init__()
        self.angles = angles
        self.distribution_type = distribution_type

    def forward(self, x: torch.Tensor | tuple[torch.Tensor],
                labels: torch.Tensor | tuple[torch.Tensor] | None = None) -> torch.Tensor | tuple[torch.Tensor, ...]:

        # TODO implement rotation
        raise NotImplementedError("Implementation todo")
        return x, labels


def Flip_coords_X(coords: torch.Tensor,
                  image_width: int
                  ) -> torch.Tensor:
    """
    Flip x-coordinates horizontally for a set of coordinates, given the image width.

    This function flips the x-coordinates of points or batches of points horizontally,
    such that the leftmost point becomes the rightmost and vice versa, relative to the image width.

    Args:
        coords (torch.Tensor): Tensor of shape (N, 2) or (B, N, 2) representing coordinates.
        image_width (int): The width of the image.

    Returns:
        torch.Tensor: Tensor of the same shape as `coords` with x-coordinates flipped.

    Raises:
        ValueError: If `coords` does not have shape (N,2) or (B,N,2).
    """

    if coords.dim() == 2:
        # Get x, y coordinates from (N,2)
        x = coords[:, 0]
        y = coords[:, 1]

        # Compute new X coordinates
        new_x = (image_width - 1) - x
        return torch.stack([new_x, y], dim=1)

    elif coords.dim() == 3:
        # Get x, y coordinates from (B,N,2)
        x = coords[..., 0]
        y = coords[..., 1]

        # Compute new X coordinates
        new_x = (image_width - 1) - x
        return torch.stack([new_x, y], dim=-1)

    else:
        raise ValueError("coords must have shape (N,2) or (B,N,2)")


def Flip_coords_Y(coords: torch.Tensor,
                  image_height: int
                  ) -> torch.Tensor:
    """
    Flip y-coordinates vertically for a set of coordinates, given the image height.

    This function flips the y-coordinates of points or batches of points vertically,
    such that the topmost point becomes the bottommost and vice versa, relative to the image height.

    Args:
        coords (torch.Tensor): Tensor of shape (N, 2) or (B, N, 2) representing coordinates.
        image_height (int): The height of the image.

    Returns:
        torch.Tensor: Tensor of the same shape as `coords` with y-coordinates flipped.

    Raises:
        ValueError: If `coords` does not have shape (N,2) or (B,N,2).
    """
    if coords.dim() == 2:
        # Get x, y coordinates from (N,2)
        x = coords[:, 0]
        y = coords[:, 1]

        # Compute new Y coords
        new_y = (image_height - 1) - y
        return torch.stack([x, new_y], dim=1)

    elif coords.dim() == 3:
        # Get x, y coordinates from (B,N,2)
        x = coords[..., 0]
        y = coords[..., 1]

        # Compute new Y coords
        new_y = (image_height - 1) - y
        return torch.stack([x, new_y], dim=-1)

    else:
        raise ValueError("coords must have shape (N,2) or (B,N,2)")


# %% Kornia augmentations module
if has_kornia:
    class CustomImageCoordsFlipHoriz(GeometricAugmentationBase2D):  # type: ignore
        def __init__(self, apply_prob:float = 0.5) -> None:
            super().__init__(apply_prob)

        def forward(self,
                    image: torch.Tensor,
                    coords: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Uses Kornia to horizontally flip a batch of images (B,C,H,W), and adjusts coords.

            Args:
                image (torch.Tensor): shape = (B, C, H, W)
                coords (torch.Tensor): shape = (B, N, 2) or (B, 2), each (x,y)
            Returns:
                flipped_image: (B, C, H, W)
                flipped_coords: (B, N, 2)
            """
            # Flip the batch of images
            flipped_image = kornia.geometry.transform.hflip(image)

            # Flip coordinates
            _, _, H, W = image.shape
            flipped_coords = Flip_coords_X(coords, image_width=W)
            return flipped_image, flipped_coords

    class CustomImageCoordsFlipVert(GeometricAugmentationBase2D):  # type: ignore
        def __init__(self, apply_prob:float = 0.5) -> None:
            super().__init__(apply_prob)

        def forward(self,
                    image: torch.Tensor,
                    coords: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Uses Kornia to vertically flip a batch of images (B,C,H,W), and adjusts coords.

            Args:
                image (torch.Tensor): shape = (B, C, H, W)
                coords (torch.Tensor): shape = (B, N, 2) or (B, 2), each (x,y)
            Returns:
                flipped_image: (B, C, H, W)
                flipped_coords: (B, N, 2)
            """
            flipped_image = kornia.geometry.transform.vflip(image)
            _, _, H, W = image.shape
            flipped_coords = Flip_coords_Y(coords, image_height=H)
            return flipped_image, flipped_coords
else:
    class CustomImageCoordsFlipHoriz(GeometricAugmentationBase2D):  # type: ignore
        def __init__(self) -> None:
            raise ImportError(
                "Kornia is not installed. Run `pip install kornia`.")

    class CustomImageCoordsFlipVert(GeometricAugmentationBase2D):  # type: ignore
        def __init__(self) -> None:
            raise ImportError(
                "Kornia is not installed. Run `pip install kornia`.")

# %% Augmentation helper configuration dataclass


@dataclass
class AugmentationConfig:
    # Input specification
    input_data_keys: list[DataKey]
    keepdim: bool = True
    same_on_batch: bool = False
    random_apply_minmax: tuple[int, int] = (1, -1)
    device: str | None = None  # Device to run augmentations on, if None, uses torch default

    # Affine roto-translation augmentation
    affine_align_corners: bool = False
    affine_fill_value: int = 0  # Fill value for empty pixels after rotation

    rotation_angle: float | tuple[float, float] = (0.0, 360.0)
    rotation_aug_prob: float = 0.0

    # Translation parameters (in pixels)
    shift_aug_prob: float = 0.0
    max_shift_img_fraction: float | tuple[float, float] = (0.5, 0.5)
    translate_distribution_type: Literal["uniform", "normal"] = "uniform"

    # Default is None = image centre
    # rotation_centre: tuple[float, float] | None = None
    # rotation_interp_mode: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR
    # rotation_expand: bool = False  # If True, expands rotated image to fill size

    # Flip augmentation probability
    hflip_prob: float = 0.0
    vflip_prob: float = 0.0

    # Optional flag to specify if image is already in the torch layout (overrides guess)
    is_torch_layout: bool | None = None

    # Poisson shot noise
    poisson_shot_noise_aug_prob: float = 0.0

    # Gaussian noise
    sigma_gaussian_noise_dn: float | tuple[float, float] = 1.0
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
    invalid_sample_remedy_action: Literal["discard", "resample", "original"] = "original"
    max_invalid_resample_attempts: int = 10  # Max attempts to resample invalid images
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


# %% Augmentation helper class
# TODO (PC) add capability to support custom augmentation module by appending it in the user-specified location ("append_custom_module_after = (module, <literal>)" that maps to a specified entry in the augs_ops list. The given module is then inserted into the list at the specified position)

class ImageAugmentationsHelper(nn.Module):
    def __init__(self, augs_cfg: AugmentationConfig):
        super().__init__()
        self.augs_cfg = augs_cfg
        self.num_aug_ops = 0

        # TODO add input_data_keys to AugmentationConfig
        # ImageSequential seems not importable from kornia

        # Define kornia augmentation pipeline
        augs_ops: list[GeometricAugmentationBase2D |
                       IntensityAugmentationBase2D] = []
        torch_vision_ops = nn.ModuleList()

        # TODO: add rotation augmentation, for simple cases, it is sufficient to rotate the image and pad with zero. Do it before translation.
        # Geometric augmentations
        # if augs_cfg.rotation_aug_prob > 0:
        #    rotation_aug = transforms.RandomRotation(degrees=augs_cfg.rotation_angle,
        #                                             center=augs_cfg.rotation_centre,
        #                                             interpolation=augs_cfg.rotation_interp_mode,
        #                                             expand=augs_cfg.rotation_expand,
        #                                             fill=augs_cfg.rotation_fill_value)
        #    torch_vision_ops.append(transforms.RandomApply([rotation_aug],
        #                                                   p=augs_cfg.rotation_aug_prob))

        # Intensity augmentations
        if augs_cfg.brightness_aug_prob > 0:
            # Random brightness scaling
            augs_ops.append(K.RandomBrightness(brightness=augs_cfg.min_max_brightness_factor,
                                               p=augs_cfg.brightness_aug_prob,
                                               keepdim=True,
                                               clip_output=False))

        if augs_cfg.contrast_aug_prob > 0:
            # Random contrast scaling
            augs_ops.append(K.RandomContrast(contrast=augs_cfg.min_max_contrast_factor,
                                             p=augs_cfg.contrast_aug_prob,
                                             keepdim=True,
                                             clip_output=False))

        if augs_cfg.gaussian_blur_aug_prob > 0:
            # Random Gaussian blur
            augs_ops.append(K.RandomGaussianBlur(kernel_size=augs_cfg.kernel_size,
                                                 sigma=augs_cfg.sigma_gaussian_blur,
                                                 p=augs_cfg.gaussian_blur_aug_prob,
                                                 keepdim=True))

        if augs_cfg.poisson_shot_noise_aug_prob > 0:
            # FIXME it seems that possion shot noise cannot be constructed, investigate
            augs_ops.append(PoissonShotNoise(
                probability=augs_cfg.poisson_shot_noise_aug_prob))

        if augs_cfg.gaussian_noise_aug_prob > 0:
            # Random Gaussian noise
            augs_ops.append(RandomGaussianNoiseVariableSigma(
                sigma_noise=augs_cfg.sigma_gaussian_noise_dn, gaussian_noise_aug_prob=augs_cfg.gaussian_noise_aug_prob))

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
            augs_ops.append(K.RandomAffine(degrees=rotation_degrees,
                                           translate=translate_shift,
                                           p=augs_cfg.rotation_aug_prob,
                                           keepdim=True,
                                           align_corners=augs_cfg.affine_align_corners,
                                           same_on_batch=False))

        # Flip augmentation
        if augs_cfg.hflip_prob > 0:
            augs_ops.append(K.RandomHorizontalFlip(p=augs_cfg.hflip_prob))

        if augs_cfg.vflip_prob > 0:
            augs_ops.append(K.RandomVerticalFlip(p=augs_cfg.vflip_prob))

        if len(augs_ops) == 0:
            print(f"{colorama.Fore.LIGHTYELLOW_EX}WARNING: No augmentations defined in augs_ops! Forward pass will not do anything if called.{colorama.Style.RESET_ALL}")
        elif len(augs_ops) == 1:
            # If len of augs_ops == 1 add placeholder augs for random_apply
            augs_ops.append(PlaceholderAugmentation())

        self.num_aug_ops = len(augs_ops)

        # Build AugmentationSequential from nn.ModuleList
        # Use maximum number of augmentations if upper bound not provided
        if augs_cfg.random_apply_minmax[1] == -1:
            random_apply_minmax_ = list(augs_cfg.random_apply_minmax)
            random_apply_minmax_[1] = len(augs_ops) - 1
        else:
            random_apply_minmax_ = list(augs_cfg.random_apply_minmax)
            random_apply_minmax_[1] = random_apply_minmax_[1] - 1

        # Transfer all modules to device if specified
        if augs_cfg.device is not None:

            for aug_op in augs_ops:
                aug_op.to(augs_cfg.device)

            for aug_op in torch_vision_ops:
                aug_op.to(augs_cfg.device)

        self.kornia_augs_module = AugmentationSequential(*augs_ops,
                                                         data_keys=augs_cfg.input_data_keys,
                                                         same_on_batch=False,
                                                         keepdim=False,
                                                         random_apply=(random_apply_minmax_[0], random_apply_minmax_[1])).to(augs_cfg.device)

        # if augs_cfg.append_custom_module_after_ is not None:
        #    pass

        self.torchvision_augs_module = nn.Sequential(
            *torch_vision_ops) if len(torch_vision_ops) > 0 else None

    # images: ndArrayOrTensor | tuple[ndArrayOrTensor, ...],
    # labels: ndArrayOrTensor

    def forward(self,
                *inputs: ndArrayOrTensor | tuple[ndArrayOrTensor, ...],
                ) -> tuple[ndArrayOrTensor | tuple[ndArrayOrTensor, ...], ndArrayOrTensor]:
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

            # Detect type, convert to torch Tensor [B,C,H,W], determine scaling factor
            is_numpy = isinstance(images_, np.ndarray)
            img_tensor, to_numpy, scale_factor = self.preprocess_images_(
                images_)

            # Undo scaling before adding augs if is_normalized
            if scale_factor is not None and self.augs_cfg.is_normalized:
                img_tensor = img_tensor * scale_factor

            # Apply torchvision augmentation module
            if self.torchvision_augs_module is not None:
                # TODO any way to modify the rotation centre at runtime? --> No way, need to switch to kornia custom with warp_affine
                # TODO how to do labels update in torchvision?
                img_tensor = self.torchvision_augs_module(img_tensor)

            # Apply translation
            # if self.augs_cfg.shift_aug_prob > 0:
            #    img_shifted, lbl_shifted = self.translate_batch_(
            #        img_tensor, labels)
            # else:
            #    img_shifted = img_tensor
            #    lbl_shifted = numpy_to_torch(labels).float()

            # Recompose inputs replacing image
            inputs = list(inputs)
            inputs[img_index] = img_tensor

            ##########
            # Unsqueeze keypoints if input is [B, 2], must be [N, 2]
            # DEVNOTE temporary code that requires extension to be more general!
            # TODO find a way to actually get keypoints and other entries without to index those manually!

            lbl_to_unsqueeze = inputs[1].dim() == 2
            if lbl_to_unsqueeze:
                # Input is (B,2) --> unsqueeze
                inputs[1] = inputs[1].unsqueeze(1)

            keypoints = inputs[1][..., :2]
            additional_entries = inputs[1][...,
                                           2:] if inputs[1].shape[-1] > 2 else None
            inputs[1] = keypoints
            ##########

            # Apply augmentations module
            if self.num_aug_ops > 0:
                aug_inputs = self.kornia_augs_module(*inputs)

                # Transformed image validator and fixer
                if self.augs_cfg.enable_batch_validation_check:
                    aug_inputs = self.validate_fix_input_img_(*aug_inputs, original_inputs=inputs)

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

            # Convert back to numpy if was ndarray
            if to_numpy is True:
                aug_inputs[img_index] = torch_to_numpy(
                    aug_inputs[img_index].permute(0, 2, 3, 1))
                aug_inputs[lbl_index] = torch_to_numpy(aug_inputs[lbl_index])

        # DEVNOTE: image appears to be transferred to cpu for no reason. To investigate.
        ###
        aug_inputs[0] = aug_inputs[0].to(keypoints.device)
        aug_inputs[1].to(keypoints.device)
        ###

        return aug_inputs

    def preprocess_images_(self,
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
            scale_factor = self.determine_scale_factor_(images)

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
            scale_factor = self.determine_scale_factor_(images)

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

    def determine_scale_factor_(self, imgs_array: ndArrayOrTensor) -> float:

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

    def validate_fix_input_img_(self, *inputs: ndArrayOrTensor | tuple[ndArrayOrTensor, ...], original_inputs : ndArrayOrTensor | tuple[ndArrayOrTensor, ...]) -> None:
        """
        Validate input images after augmentation. Attempt fix according to selected remedy action.
        """
        # Determine validity of input images according to is_valid_image_ criteria (default or custom)
        # TODO: allow _is_valid_image to be custom by overloading with user-specified function returning a mask of size (B, N). Use a functor to enforce constraints on the method signature
        # TODO: requires extensive testing!

        inputs = list(inputs)
        img_index = self.augs_cfg.input_data_keys.index(DataKey.IMAGE)
        lbl_index = self.augs_cfg.input_data_keys.index(DataKey.KEYPOINTS) # TODO (PC) absolutely requires extension!

        # Scan for invalid samples
        is_valid_mask, invalid_samples = self._is_valid_image(*inputs)
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
                invalid_original = [input[~is_valid_mask] for input in original_inputs]

                while not_all_valid:
                    
                    # Rerun augs module on invalid samples
                    new_aug_inputs_ = self.kornia_augs_module(*invalid_original)

                    # Check new validity mask
                    is_valid_mask_tmp, _ = self._is_valid_image(*new_aug_inputs_)

                    not_all_valid = not(is_valid_mask_tmp.all())

                    if iter_counter == self.augs_cfg.max_invalid_resample_attempts:
                        raise RuntimeError(f"Max invalid resample attempts reached: {iter_counter}. Augmentation helper is not able to provide a fully valid batch. Please verify your augmentation configuration.")
                    
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

    def _is_valid_image(self, *inputs : ndArrayOrTensor | tuple[ndArrayOrTensor, ...]):
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
            invalid_inputs (tuple): Tuple of tensors containing only the invalid samples.
        """

        # Compute mean across channels and spatial dims
        img_index = self.augs_cfg.input_data_keys.index(DataKey.IMAGE)

        B = inputs[img_index].shape[0]
        #mean_per_image = torch.abs(inputs[img_index]).mean(dim=(1, 2, 3))  # (B,)
        
        # Move all inputs to the same device
        inputs = tuple(input.to(self.augs_cfg.device) for input in inputs)

        # A threshold to detect near-black images (tune if needed)
        is_pixel_bright_count_mask = (torch.abs(inputs[img_index]) > 1).view(B, -1).sum(dim=1)
        is_valid_mask = is_pixel_bright_count_mask >= self.augs_cfg.min_num_bright_pixels

        # Indices of invalid images
        invalid_indices = (~is_valid_mask).nonzero(as_tuple=True)[0]

        # Select invalid samples
        invalid_inputs = tuple(torch.index_select(input, dim=0, index=invalid_indices) for input in inputs)

        # Return validity mask and new invalid tensor samples (using index_select)
        return is_valid_mask, invalid_inputs


    # TODO (PC) move method to dedicated custom augmentation class
    # DEPRECATED, move outside for archive
    def translate_batch_(self,
                         images: torch.Tensor,
                         labels: ndArrayOrTensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """
            images: [B,C,H,W]
            labels: torch.Tensor[B,N] or np.ndarray
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
        if isinstance(self.augs_cfg.max_shift_img_fraction, (tuple, list)):
            max_x, max_y = self.augs_cfg.max_shift_img_fraction
        else:
            max_x, max_y = (self.augs_cfg.max_shift_img_fraction,
                            self.augs_cfg.max_shift_img_fraction)

        if self.augs_cfg.translate_distribution_type == "uniform":
            # Sample shifts by applying 0.99 margin
            dx = torch.randint(-int(max_x), int(max_x)+1, (B,))
            dy = torch.randint(-int(max_y), int(max_y)+1, (B,))
        elif self.augs_cfg.translate_distribution_type == "normal":
            # Sample shifts from normal distribution
            dx = torch.normal(mean=0.0, std=max_x, size=(B,))
            dy = torch.normal(mean=0.0, std=max_y, size=(B,))
        else:
            raise ValueError(
                f"Unsupported distribution type: {self.translate_distribution_type}. Supported types are 'uniform' and 'normal'.")

        shifted_imgs = images.new_zeros(images.shape)

        # TODO improve this method, currently not capable of preventing the object to exit the plane
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

    # Overload "to" method
    def to(self, *args, **kwargs):
        """Overload to method to apply to all submodules."""
        super().to(*args, **kwargs)
        self.kornia_augs_module.to(*args, **kwargs)

        if self.torchvision_augs_module is not None:
            self.torchvision_augs_module.to(*args, **kwargs)

        return self

# %% Prototypes TODO


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


# TODO GeometryAugsModule TBD may be unneeded
class GeometryAugsModule(AugsBaseClass):
    def __init__(self):
        super(GeometryAugsModule, self).__init__()

        # Example usage
        self.augmentations = AugmentationSequential(
            kornia_aug.RandomRotation(degrees=30.0, p=1.0),
            kornia_aug.RandomAffine(degrees=0, translate=(0.1, 0.1), p=1.0),
            data_keys=["input", "mask"]
        )  # Define the keys: image is "input", mask is "mask"

    def forward(self, x: torch.Tensor, labels: torch.Tensor | tuple[torch.Tensor]) -> torch.Tensor:
        # TODO define interface (input, output format and return type)
        x, labels = self.augmentations(x, labels)

        return x, labels


############################################################################################################################
# %% DEPRECATED functions (legacy code)
def build_kornia_augs(sigma_noise: float, sigma_gaussian_blur: tuple | float = (0.0001, 1.0),
                      brightness_factor: tuple | float = (0.0001, 0.01),
                      contrast_factor: tuple | float = (0.0001, 0.01)) -> torch.nn.Sequential:

    # Define kornia augmentation pipeline

    # Random brightness
    brightness_min, brightness_max = brightness_factor if isinstance(
        brightness_factor, tuple) else (brightness_factor, brightness_factor)

    random_brightness = kornia_aug.RandomBrightness(brightness=(
        brightness_min, brightness_max), clip_output=False, same_on_batch=False, p=1.0, keepdim=True)

    # Random contrast
    contrast_min, contrast_max = contrast_factor if isinstance(
        contrast_factor, tuple) else (contrast_factor, contrast_factor)

    random_contrast = kornia_aug.RandomContrast(contrast=(
        contrast_min, contrast_max), clip_output=False, same_on_batch=False, p=1.0, keepdim=True)

    # Gaussian Blur
    sigma_gaussian_blur_min, sigma_gaussian_blur_max = sigma_gaussian_blur if isinstance(
        sigma_gaussian_blur, tuple) else (sigma_gaussian_blur, sigma_gaussian_blur)
    gaussian_blur = kornia_aug.RandomGaussianBlur(
        (5, 5), (sigma_gaussian_blur_min, sigma_gaussian_blur_max), p=0.75, keepdim=True)

    # Gaussian noise
    gaussian_noise = kornia_aug.RandomGaussianNoise(
        mean=0.0, std=sigma_noise, p=0.75, keepdim=True)

    # Motion blur
    # direction_min, direction_max = -1.0, 1.0
    # motion_blur = kornia_aug.RandomMotionBlur((3, 3), (0, 360), direction=(direction_min, direction_max), p=0.75, keepdim=True)

    return torch.nn.Sequential(random_brightness, random_contrast, gaussian_blur, gaussian_noise)


def TranslateObjectImgAndPoints(image: torch.Tensor,
                                label: torch.Tensor,
                                max_size_in_pix: float | torch.Tensor | list[float]) -> tuple:

    if not (isinstance(max_size_in_pix, torch.Tensor)):
        max_size_in_pix = torch.Tensor([max_size_in_pix, max_size_in_pix])

    num_entries = 1  # TODO update to support multiple images

    # Get image size
    image_size = image.shape

    # Get max shift coefficients (how many times the size enters half image with margin)
    # TODO validate computation
    max_vertical = 0.99 * (0.5 * image_size[1] / max_size_in_pix[1] - 1)
    max_horizontal = 0.99 * (0.5 * image_size[2] / max_size_in_pix[0] - 1)

    raise NotImplementedError("TODO")

    # Sample shift interval uniformly --> TODO for batch processing: this has to generate uniformly sampled array
    shift_horizontal = torch.randint(-max_horizontal,
                                     max_horizontal, (num_entries,))
    shift_vertical = torch.randint(-max_vertical, max_vertical, (num_entries,))

    # Shift vector --> TODO for batch processing: becomes a matrix
    origin_shift_vector = torch.round(torch.Tensor(
        [shift_horizontal, shift_vertical]) * max_size_in_pix)

    # print("Origin shift vector: ", originShiftVector)

    # Determine index for image cropping
    # Vertical
    idv1 = int(np.floor(np.max([0, origin_shift_vector[1]])))
    idv2 = int(
        np.floor(np.min([image_size[1], origin_shift_vector[1] + image_size[1]])))

    # Horizontal
    idu1 = int(np.floor(np.max([0, origin_shift_vector[0]])))
    idu2 = int(
        np.floor(np.min([image_size[2], origin_shift_vector[0] + image_size[2]])))

    croppedImg = image[:, idv1:idv2, idu1:idu2]

    # print("Cropped image shape: ", croppedImg.shape)

    # Create new image and store crop
    shiftedImage = torch.zeros(
        image_size[0], image_size[1], image_size[2], dtype=torch.float32)

    # Determine index for pasting
    # Vertical
    idv1 = int(abs(origin_shift_vector[1])
               ) if origin_shift_vector[1] < 0 else 0
    idv2 = idv1 + croppedImg.shape[1]
    # Horizontal
    idu1 = int(abs(origin_shift_vector[0])
               ) if origin_shift_vector[0] < 0 else 0
    idu2 = idu1 + croppedImg.shape[2]

    shiftedImage[:, idv1:idv2, idu1:idu2] = croppedImg

    # Shift labels (note that coordinate of centroid are modified in the opposite direction as of the origin)
    shiftedLabel = label - \
        torch.Tensor(
            [origin_shift_vector[0], origin_shift_vector[1], 0], dtype=torch.float32)

    return shiftedImage, shiftedLabel


# %% DEVELOPMENT CODE
if __name__ == "__main__":
    pass

"""
    # For possible later development: translation on batch, tensorized
    def _translate_batch(self,
                         images: torch.Tensor,
                         labels: ArrayOrTensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        B,C,H,W = images.shape
        # convert labels
        lbl = torch.from_numpy(labels).float() if isinstance(labels, np.ndarray) else labels.float()
        if lbl.dim()==2:
            lbl = lbl.unsqueeze(0)
        # prepare per-sample max shifts
        if isinstance(self.cfg.max_shift, torch.Tensor):
            ms = self.cfg.max_shift.to(images.device).long()
        elif isinstance(self.cfg.max_shift, list):
            ms = torch.tensor(self.cfg.max_shift, device=images.device).long()
        else:
            fx,fy = self.cfg.max_shift if isinstance(self.cfg.max_shift,tuple) else (self.cfg.max_shift,self.cfg.max_shift)
            ms = torch.tensor([[int(fx),int(fy)]]*B, device=images.device)
        # random shifts uniform integer in [-max, max]
        dx = torch.randint(-ms[:,0], ms[:,0]+1, (B,), device=images.device)
        dy = torch.randint(-ms[:,1], ms[:,1]+1, (B,), device=images.device)
        # normalized translation for affine: tx = dx/(W/2), ty = dy/(H/2)
        tx = dx.float()/(W/2)
        ty = dy.float()/(H/2)
        # build theta for each sample: [[1,0,tx],[0,1,ty]]
        theta = torch.zeros((B,2,3), device=images.device, dtype=images.dtype)
        theta[:,0,0] = 1; theta[:,1,1] = 1
        theta[:,0,2] = tx; theta[:,1,2] = ty
        # grid and sample
        grid = F.affine_grid(theta, images.size(), align_corners=False)
        shifted = F.grid_sample(images, grid, padding_mode='zeros', align_corners=False)
        # shift labels in pixel space
        lbl_t = lbl - torch.stack([dx,dy], dim=1).unsqueeze(1).float()
"""
