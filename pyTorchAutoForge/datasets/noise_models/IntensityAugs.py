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


class RandomBinarizeImage(IntensityAugmentationBase2D):
    def __init__(self,
                 aug_prob: float = 0.5,
                 masking_quantile: float = 0.15) -> None:
        super().__init__(p=aug_prob)
        self.masking_quantile = masking_quantile

    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict[str, torch.Tensor],
                        flags: dict[str, Any],
                        transform: torch.Tensor | None = None) -> torch.Tensor:

        img_batch = input[0]
        B = input.shape[0]
        device, dtype = img_batch.device, img_batch.dtype

        # Compute threshold
        avg_brightness = img_batch.mean(dim=1, keepdim=True)  # B×1×H×W
        brightness_thr = (avg_brightness.view(
            B, 1, -1)).quantile(self.masking_quantile, dim=2, keepdim=True)

        # Get binarized image
        img_batch_mask = (avg_brightness > brightness_thr).float()

        return img_batch_mask


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
