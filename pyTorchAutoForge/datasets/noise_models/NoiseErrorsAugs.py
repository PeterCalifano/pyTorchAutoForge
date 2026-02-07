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
                        params: dict[str, torch.Tensor],
                        flags: dict[str, Any],
                        transform: torch.Tensor | None = None) -> torch.Tensor:
        """
        Applies Poisson shot noise to the input batch of images.

        Args:
            x (torch.Tensor | tuple[torch.Tensor]): Input images as a tensor or tuple of tensors.
            labels (torch.Tensor | tuple[torch.Tensor] | None, optional): Optional labels associated with the images.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: Images with Poisson shot noise applied, or a tuple containing such images.

        """
        # Randomly sample a boolean mask to index batch size
        B = input.shape[0]
        device, dtype = input.device, input.dtype

        # Pixel value is the variance of the Photon Shot Noise (higher where brighter).
        # Therefore, the mean rate parameter mu is equal to the DN at the specific pixel.
        photon_shot_noise = torch.poisson(input)

        # Sum noise to the original images
        return input + photon_shot_noise

    def generate_parameters(self,
                            shape: tuple[int, ...]) -> dict:
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
                 sigma_noise: int | float | tuple[float | int, float | int],
                 gaussian_noise_aug_prob: float = 0.5,
                 keep_scalar_sigma_fixed: bool = False,
                 enable_img_validation_mode: bool = True,
                 validation_min_num_bright_pixels: int = 50,
                 validation_pixel_threshold: float = 5.0):

        # Init superclass
        super().__init__(p=gaussian_noise_aug_prob)

        # Store parameters
        self.sigma_gaussian_noise_dn = sigma_noise
        self.keep_scalar_sigma_fixed: bool = keep_scalar_sigma_fixed

        self.enable_img_validation_mode: bool = enable_img_validation_mode
        self.validation_min_num_bright_pixels: int = validation_min_num_bright_pixels

        # Pixel intensity threshold to consider a pixel "bright"
        self.validation_pixel_threshold: float | Tensor = validation_pixel_threshold

        # Initialization checks on sigma values
        if isinstance(self.sigma_gaussian_noise_dn, (tuple, list)):
            if len(self.sigma_gaussian_noise_dn) != 2:
                raise ValueError(
                    f'Invalid sigma_noise tuple length. Expected 2 got {len(self.sigma_gaussian_noise_dn)}')
            if self.sigma_gaussian_noise_dn[0] < 0 or self.sigma_gaussian_noise_dn[1] < 0:
                raise ValueError(
                    'Sigma noise values must be non-negative.')
            if self.sigma_gaussian_noise_dn[0] >= self.sigma_gaussian_noise_dn[1]:
                raise ValueError(
                    'Invalid sigma noise range. Minimum must be less than maximum.')
            # Cast to tuple to ensure immutability
            self.sigma_gaussian_noise_dn = tuple(
                self.sigma_gaussian_noise_dn)  # type:ignore

        elif isinstance(self.sigma_gaussian_noise_dn, (int, float)):
            if self.sigma_gaussian_noise_dn < 0:
                raise ValueError('Sigma noise value must be non-negative.')
            # Cast to float
            self.sigma_gaussian_noise_dn = float(self.sigma_gaussian_noise_dn)

        else:
            raise TypeError(
                f'Invalid sigma_noise value. Expected int, float or tuple[float | int, float | int] got {type(self.sigma_gaussian_noise_dn)}')

    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict[str, torch.Tensor],
                        flags: dict[str, Any],
                        transform: torch.Tensor | None = None) -> torch.Tensor:

        # Expect BCHW
        if input.ndim != 4:
            raise ValueError(
                f"Expected 4D BCHW, got shape {tuple(input.shape)}")

        B, C, H, W = input.shape
        device = input.device

        # Determine sigma per sample
        sigma_val = self.sigma_gaussian_noise_dn

        if isinstance(sigma_val, (tuple, list)):
            # Get min-max values
            min_s, max_s = sigma_val

            # Do uniform sampling per sample
            sigma_array = (max_s - min_s) * \
                torch.rand(B, device=device) + min_s

        elif isinstance(sigma_val, (float, int)):

            if self.keep_scalar_sigma_fixed:
                # Use same assigned sigma as fixed value
                sigma_array = torch.full((B,), float(sigma_val), device=device)

            else:
                # Randomize using interval [0, sigma_val]
                sigma_array = (sigma_val * torch.rand(B, device=device))
        else:
            raise TypeError(
                f'Invalid sigma_noise value. Expected int, float or tuple[float | int, float | int] got {type(sigma_val)}')

        # If validation mode, check image content first. Set sigma to zero for those entries before applying
        if self.enable_img_validation_mode:

            # Check threshold validity (auto-correct based on maximum value in images)
            max_img_value = input.abs().max()
            flat_input = input.abs().reshape(B, -1)

            if self.validation_pixel_threshold >= max_img_value:

                print(f"{colorama.Fore.LIGHTYELLOW_EX}WARNING: validation_pixel_threshold ({self.validation_pixel_threshold}) is greater than or equal to the maximum image pixel value ({max_img_value}). Adjusting threshold it automatically to 0.05 * median of max per sample: {self.validation_pixel_threshold}.{colorama.Style.RESET_ALL}")

                # Get max per sample
                max_per_sample = flat_input.max(dim=1).values

                # Compute as 0.05 of median of max image values
                validation_pixel_threshold_ = 0.05 * max_per_sample
            else:
                # Use the configured scalar threshold for all samples
                validation_pixel_threshold_ = torch.full((B,),
                                                         float(
                    self.validation_pixel_threshold),
                    device=input.device,
                    dtype=flat_input.dtype,
                )

            # Count number of bright pixels per image in batch
            is_pixel_bright_count_mask = flat_input.gt(validation_pixel_threshold_.view(
                B, 1)).sum(dim=1)  # Get count of bright pixels per image

            # Determine valid images mask
            # Get bool mask where >= min_bright pixels shapes as (B,)
            is_valid_mask = is_pixel_bright_count_mask.ge(
                self.validation_min_num_bright_pixels)

            # Set sigma to zero for invalid images by multiplying with bool mask
            sigma_array = sigma_array * is_valid_mask.to(sigma_array.dtype)

        # Sample image noise and apply to input batch
        sigma_array = sigma_array.view(B, 1, 1, 1)
        noise = torch.randn_like(input) * sigma_array

        return input + noise
