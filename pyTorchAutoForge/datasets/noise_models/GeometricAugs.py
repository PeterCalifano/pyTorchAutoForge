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

# %% Geometric augmentations
class BorderAwareRandomAffine(GeometricAugmentationBase2D):
    """
    BorderAwareRandomAffine Class reimplementing Kornia's RandomAffine with border crossing constraints.

    This augmentation applies random affine transformations to a batch of images while considering border crossing constraints. It detects bright pixel runs along image borders to classify crossing types and modifies the affine transformation parameters accordingly before computing the transformation. The rest of the implementation follows RandomAffine. Depending on the detected crossing type, it constrains translations and rotations to prevent unrealistic augmentations. Detection is performed based on bright pixel runs along image borders (tailored for typical space images with near black background).

    :param GeometricAugmentationBase2D: Base class for 2D geometric augmentations in Kornia.
    :type GeometricAugmentationBase2D: class
    """

    def __init__(self,
                 base_random_affine: K.RandomAffine,
                 num_pix_crossing_detect: int = 10,
                 intensity_threshold_uint8: float = 7.0):
        super().__init__(p=base_random_affine.p,
                         same_on_batch=base_random_affine.same_on_batch,
                         keepdim=base_random_affine.keepdim)

        # Input validity check
        assert isinstance(
            base_random_affine, K.RandomAffine), "base_random_affine must be an instance of K.RandomAffine"

        # Store base random affine
        self.base_random_affine = base_random_affine

        # Get generator and flags from RandomAffine base
        self._param_generator = base_random_affine._param_generator
        self.flags = dict(resample=base_random_affine.flags["resample"],
                          padding_mode=base_random_affine.flags["padding_mode"],
                          align_corners=base_random_affine.flags["align_corners"])

        # Store attributes
        self.num_pix_crossing_detect = num_pix_crossing_detect
        self.intensity_threshold_uint8 = intensity_threshold_uint8

        # Assert validity of parameters
        if self.num_pix_crossing_detect <= 0:
            raise ValueError(
                "num_pix_crossing_detect must be a positive integer.")
        if self.intensity_threshold_uint8 < 0:
            raise ValueError(
                "intensity_threshold_uint8 must be a non-negative float.")

    def compute_transformation(self,
                               input: torch.Tensor,
                               params: dict[str, torch.Tensor],
                               flags: dict[str, Any]) -> torch.Tensor:
        """
        compute_transformation Computes the affine transformation matrix for each sample in the batch, taking into account border crossing constraints. Detection is performed based on bright pixel runs along image borders (tailored for typical space images with near black background).
        """

        # Get transform parameters (per sample)
        translations = params["translations"]
        angle = params["angle"]
        scale = params["scale"]
        shear_x = params["shear_x"]
        shear_y = params["shear_y"]

        # Get previously computed crossing type for this batch if available
        crossing_types = params.get("crossing_types", None)

        # Else compute crossing_types mask
        if crossing_types is None:
            crossing_types, _, _ = BorderAwareRandomAffine.Detect_border_crossing(
                img_batch=input,
                num_pix_detect=self.num_pix_crossing_detect,
                intensity_threshold_uint8=self.intensity_threshold_uint8)
            params["crossing_types"] = crossing_types

        # Determine masks for transformations
        crossing_types = crossing_types.to(device=translations.device)

        vertical_only = crossing_types == 1
        horizontal_only = crossing_types == 2
        blocked = crossing_types == 3

        # If any crossing found, modify params accordingly
        if vertical_only.any() or horizontal_only.any() or blocked.any():

            # Clone to avoid modifying original params
            translations = translations.clone()
            angle = angle.clone()
            scale = scale.clone()
            shear_x = shear_x.clone()
            shear_y = shear_y.clone()

            # Apply vertical only translation constraint (null out X translation)
            if vertical_only.any():
                translations[vertical_only, 0] = 0
                angle[vertical_only] = 0
                shear_x[vertical_only] = 0
                shear_y[vertical_only] = 0
                scale[vertical_only] = 1

            # Apply horizontal only translation constraint (null out Y translation)
            if horizontal_only.any():
                translations[horizontal_only, 1] = 0
                angle[horizontal_only] = 0
                shear_x[horizontal_only] = 0
                shear_y[horizontal_only] = 0
                scale[horizontal_only] = 1

            # Apply blocked constraint (null out all transformations)
            if blocked.any():
                translations[blocked] = 0
                angle[blocked] = 0
                shear_x[blocked] = 0
                shear_y[blocked] = 0
                scale[blocked] = 1

            # Re-assign modified params
            params["translations"] = translations
            params["angle"] = angle
            params["scale"] = scale
            params["shear_x"] = shear_x
            params["shear_y"] = shear_y

        return KG.get_affine_matrix2d(
            torch.as_tensor(params["translations"],
                            device=input.device, dtype=input.dtype),
            torch.as_tensor(params["center"],
                            device=input.device, dtype=input.dtype),
            torch.as_tensor(params["scale"],
                            device=input.device, dtype=input.dtype),
            torch.as_tensor(params["angle"],
                            device=input.device, dtype=input.dtype),
            KG.deg2rad(torch.as_tensor(
                params["shear_x"], device=input.device, dtype=input.dtype)),
            KG.deg2rad(torch.as_tensor(
                params["shear_y"], device=input.device, dtype=input.dtype)),
        )

    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict[str, torch.Tensor],
                        flags: dict[str, Any],
                        transform: torch.Tensor | None = None) -> torch.Tensor:

        _, _, height, width = input.shape

        if not isinstance(transform, torch.Tensor):
            raise TypeError(
                f"Expected the `transform` be a Tensor. Got {type(transform)}.")

        # Apply affine warp
        return KG.warp_affine(input,
                              transform[:, :2, :],
                              (height, width),
                              flags["resample"].name.lower(),
                              align_corners=flags["align_corners"],
                              padding_mode=flags["padding_mode"].name.lower(),
                              )

    def inverse_transform(self,
                          input: torch.Tensor,
                          flags: dict[str, Any],
                          transform: torch.Tensor | None = None,
                          size: tuple[int, int] | None = None) -> torch.Tensor:

        if not isinstance(transform, torch.Tensor):
            raise TypeError(
                f"Expected the `transform` be a Tensor. Got {type(transform)}.")

        return self.apply_transform(input,
                                    params=self._params,
                                    transform=torch.as_tensor(
                                        transform,
                                        device=input.device,
                                        dtype=input.dtype),            flags=flags,
                                    )

    @staticmethod
    def Make_constrained_random_affines(base: K.RandomAffine) -> tuple[K.RandomAffine, K.RandomAffine, K.RandomAffine]:
        """
        Make_constrained_random_affines _summary_

        _extended_summary_

        :param base: _description_
        :type base: K.RandomAffine
        :return: _description_
        :rtype: tuple[RandomAffine, RandomAffine, RandomAffine]
        """
        gen = base._param_generator
        tx, ty = (0.0, 0.0) if gen.translate is None else [
            float(v) for v in gen.translate]

        common_opts = dict(resample=base.flags["resample"],
                           padding_mode=base.flags["padding_mode"],
                           same_on_batch=base.same_on_batch,
                           align_corners=base.flags["align_corners"],
                           p=base.p,
                           keepdim=base.keepdim,
                           )

        # Build constrained random affines
        vertical_only = K.RandomAffine(degrees=0.0,
                                       translate=(0.0, ty),
                                       scale=None,
                                       shear=None,
                                       **common_opts)

        horizontal_only = K.RandomAffine(degrees=0.0,
                                         translate=(tx, 0.0),
                                         scale=None,
                                         shear=None, **common_opts)

        return base, vertical_only, horizontal_only

    @staticmethod
    def Detect_border_crossing(img_batch,
                               num_pix_detect: int = 10,
                               intensity_threshold_uint8: float = 7.0):
        """
        Detect bright pixel runs along image borders to classify crossing types.

        Returns:
            crossing_type (torch.Tensor): Long tensor (B,) where 0=full/none, 1=vertical-only, 2=horizontal-only, 3=both.
            vertical_crossing (torch.Tensor): Bool tensor (B,) true when left/right borders have a run.
            horizontal_crossing (torch.Tensor): Bool tensor (B,) true when top/bottom borders have a run.
        """
        if img_batch.ndim != 4:
            raise ValueError(
                f"Expected BCHW tensor, got shape {tuple(img_batch.shape)}")

        # Create bright pixel mask based on threshold (assumes uint8 input)
        binarized_img_mask = (img_batch > intensity_threshold_uint8).any(dim=1)

        B, H, W = binarized_img_mask.shape
        device = binarized_img_mask.device

        horizontal_edges = torch.cat(
            (binarized_img_mask[:, 0, :], binarized_img_mask[:, -1, :]), dim=0)
        vertical_edges = torch.cat(
            (binarized_img_mask[:, :, 0], binarized_img_mask[:, :, -1]), dim=0)

        # Determine crossings using convolutional count
        horizontal_crossing = BorderAwareRandomAffine.ConvCount_intensity_line1d(
            horizontal_edges, num_pix_detect)
        vertical_crossing = BorderAwareRandomAffine.ConvCount_intensity_line1d(
            vertical_edges, num_pix_detect)

        # Get batch-wise masks
        horizontal_crossing = horizontal_crossing[:B] | horizontal_crossing[B:]
        vertical_crossing = vertical_crossing[:B] | vertical_crossing[B:]

        # Determine crossing type
        crossing_type = torch.full(
            (B, 1), 0, device=device, dtype=torch.int64)  # 0: all allowed
        crossing_type[vertical_crossing & ~
                      horizontal_crossing] = 1  # 1: vertical only

        crossing_type[horizontal_crossing & ~
                      vertical_crossing] = 2  # 2: horizontal only

        crossing_type[vertical_crossing &
                      horizontal_crossing] = 3  # 3: none allowed

        return crossing_type.squeeze(-1), vertical_crossing, horizontal_crossing

    @staticmethod
    def ConvCount_intensity_line1d(line_bool_mask: torch.Tensor,
                                   window_length: int,
                                   device: torch.device | str | None = None) -> torch.Tensor:
        """
        ConvCount_intensity_line1d _summary_

        _extended_summary_

        :param line_bool_mask: _description_
        :type line_bool_mask: torch.Tensor
        :param window_length: _description_
        :type window_length: int
        :param device: _description_, defaults to None
        :type device: torch.device | str | None, optional
        :return: _description_
        :rtype: torch.Tensor
        """

        # Get length of line
        length = line_bool_mask.shape[-1]

        if device is None:
            device = line_bool_mask.device

        if length < window_length:
            return torch.zeros(line_bool_mask.shape[0],
                               device=device,
                               dtype=torch.bool)

        # Define convolution kernel
        kernel = torch.ones((1, 1, window_length),
                            device=device,
                            dtype=torch.float32)

        # Run convolution across line
        conv = F.conv1d(input=line_bool_mask.float().unsqueeze(1),
                        weight=kernel)

        # Check if any window is fully filled
        return conv.amax(dim=-1) >= window_length
