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
from pyTorchAutoForge.datasets.DataAugmentation import AugsBaseClass

# %% Intensity augmentations


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
                 randomization_prob: float = 0.6,
                 masking_quantile: float = 0.85) -> None:
        super().__init__(p=noise_texture_aug_prob)

        self.randomization_prob = randomization_prob
        self.masking_quantile = masking_quantile

    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict[str, torch.Tensor],
                        flags: dict[str, Any],
                        transform: torch.Tensor | None = None) -> torch.Tensor:

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


class RandomSoftBinarizeImage(IntensityAugmentationBase2D):
    def __init__(self,
                 aug_prob: float = 0.5,
                 masking_quantile: float = 0.01,
                 blending_factor_minmax: None | tuple[float, float] = (0.3, 0.7)):
        super().__init__(p=aug_prob)
        self.masking_quantile = masking_quantile
        self.blending_factor_minmax = blending_factor_minmax

        # Assert blending factor minmax validity
        if blending_factor_minmax is not None:
            assert blending_factor_minmax[0] >= 0.0 and blending_factor_minmax[1] <= 1.0, \
                "Blending factor minmax values must be in [0, 1] range."
            assert blending_factor_minmax[0] < blending_factor_minmax[1], \
                "Blending factor min must be smaller than blending factor max."

        # Assert validity of masking quantile
        assert masking_quantile >= 0.0 and masking_quantile <= 1.0, \
            "Masking quantile must be in [0, 1] range."

    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict[str, torch.Tensor],
                        flags: dict[str, Any],
                        transform: torch.Tensor | None = None) -> torch.Tensor:

        img_batch = input.contiguous()
        B = input.shape[0]
        device, dtype = img_batch.device, img_batch.dtype

        # Convert rgb to gray if needed
        gray_brightness = img_batch
        if img_batch.shape[1] == 3:
            gray_brightness = 0.2989 * img_batch[:, 0:1, :, :] + \
                0.5870 * img_batch[:, 1:2, :, :] + \
                0.1140 * img_batch[:, 2:3, :, :]

        # Compute brightness threshold for each image in the batch based on masking quantile (illuminate regions only)
        gray_flat = gray_brightness.view(B, 1, -1)  # [B,1,H*W]
        
        # Determine validity threshold for quantile based on scale
        validity_thr = 0.0
        if torch.abs(gray_flat).max() > 1.0:
            validity_thr = 1.0  # [B,1,H*W]

        # Apply thresholding
        valid_flat = gray_flat >= validity_thr  # [B,1,H*W]

        flat_valid = torch.where(valid_flat.view(B, 1, -1),
                                 gray_flat,
                                 torch.nan)

        brightness_thr = torch.nanquantile(flat_valid,
                                           self.masking_quantile,
                                           dim=2,
                                           keepdim=True).view(B, 1, 1, 1)
        
        high_quantile_per_sample = torch.nanquantile(flat_valid,
                                           0.999,
                                           dim=2,
                                           keepdim=True).view(B, 1, 1, 1)

        # Get binarized image
        img_batch_mask = (gray_brightness >
                          brightness_thr).float().to(device, dtype)        

        # Check if blending is to be applied (if blending factors are not 0)
        if self.blending_factor_minmax is not None:
            # Sample random blending factors for each image in the batch
            blending_factors = (torch.ones(size=(B, 1, 1, 1),
                                           device=device,
                                           dtype=dtype)).uniform_(self.blending_factor_minmax[0], self.blending_factor_minmax[1])

            # Blend original gray brightness with binarized mask (keep zero regions as zero)
            return img_batch_mask * (blending_factors * high_quantile_per_sample * img_batch_mask + (1.0 - blending_factors) * gray_brightness)
        
        else:
            return img_batch_mask * high_quantile_per_sample


class RandomConvertTextureToShadingLaw(IntensityAugmentationBase2D):
    """
    RandomConvertTextureToShadingLaw Random augmentation that replaces textured regions with smooth shading based on selected shading law.

    _extended_summary_

    :param IntensityAugmentationBase2D: _description_
    :type IntensityAugmentationBase2D: _type_
    """

    def __init__(self,
                 aug_prob: float = 0.5):
        super().__init__(p=aug_prob)

    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict[str, torch.Tensor],
                        flags: dict[str, Any],
                        transform: torch.Tensor | None = None) -> torch.Tensor:
        pass
