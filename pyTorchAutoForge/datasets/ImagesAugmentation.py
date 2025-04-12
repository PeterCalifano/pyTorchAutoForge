# TODO understand how to use for labels processing
from kornia.augmentation import AugmentationSequential
import albumentations
from regex import E
import torch
from kornia import augmentation as kornia_aug
from torch import nn
from abc import ABC, abstractmethod
import pytest  # For unit tests
from dataclasses import dataclass
# , cupy # cupy for general GPU acceleration use (without torch) to test
import numpy as np
from enum import Enum


def TranslateObjectImgAndPoints(image: torch.Tensor,
                                label: torch.Tensor,
                                max_size_in_pix: float | torch.Tensor | list[float]) -> tuple:

    if not (isinstance(max_size_in_pix, torch.Tensor)):
        max_size_in_pix = torch.Tensor([max_size_in_pix, max_size_in_pix])

    num_entries = 1 # TODO update to support multiple images

    # Get image size
    image_size = image.shape

    # Get max shift coefficients (how many times the size enters half image with margin)
    # TODO validate computation
    max_vertical = 0.99 * (0.5 * image_size[1] / max_size_in_pix[1] - 1)
    max_horizontal = 0.99 * (0.5 * image_size[2] / max_size_in_pix[0] - 1)

    raise NotImplementedError("TODO")

    # Sample shift interval uniformly --> TODO for batch processing: this has to generate uniformly sampled array
    shift_horizontal = torch.randint(-max_horizontal, max_horizontal, (num_entries,))
    shift_vertical = torch.randint(-max_vertical, max_vertical, (num_entries,))


    # Shift vector --> TODO for batch processing: becomes a matrix
    origin_shift_vector = torch.round(torch.Tensor([shift_horizontal, shift_vertical]) * max_size_in_pix)

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
    idv1 = int(abs(origin_shift_vector[1])) if origin_shift_vector[1] < 0 else 0
    idv2 = idv1 + croppedImg.shape[1]
    # Horizontal
    idu1 = int(abs(origin_shift_vector[0])) if origin_shift_vector[0] < 0 else 0
    idu2 = idu1 + croppedImg.shape[2]

    shiftedImage[:, idv1:idv2, idu1:idu2] = croppedImg

    # Shift labels (note that coordinate of centroid are modified in the opposite direction as of the origin)
    shiftedLabel = label - \
        torch.Tensor(
            [origin_shift_vector[0], origin_shift_vector[1], 0], dtype=torch.float32)

    return shiftedImage, shiftedLabel


class ImageNormalizationType(Enum):
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

    def __init__(self, normalization_type: ImageNormalizationType = ImageNormalizationType.NONE):
        self.normalization_type = normalization_type

    def __call__(self, image: torch.Tensor) -> torch.Tensor:

        # Check image datatype, if not float, normalize using dtype
        if image.dtype != torch.float32 and image.dtype != torch.float64:

            # Get datatype and select normalization
            if self.normalization_type == ImageNormalizationType.SOURCE and self.normalization_type is not ImageNormalizationType.NONE:

                if image.dtype == torch.uint8:
                    self.normalization_type = ImageNormalizationType.UINT8

                elif image.dtype == torch.uint16:
                    self.normalization_type = ImageNormalizationType.UINT16

                elif image.dtype == torch.uint32:
                    self.normalization_type = ImageNormalizationType.UINT32
                else:
                    raise ValueError(
                        "Normalization type selected as SOURCE but image type is not uint8, uint16 or uint32. Cannot determine normalization value")

        # Normalize image to range [0,1]
        if self.normalization_type.value < 0.0:
            raise ValueError(
                "Normalization for images value cannot be negative.")

        return image / self.normalization_type.value


def build_kornia_augs(sigma_noise: float, sigma_blur: tuple | float = (0.0001, 1.0),
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
    sigma_blur_min, sigma_blur_max = sigma_blur if isinstance(
        sigma_blur, tuple) else (sigma_blur, sigma_blur)
    gaussian_blur = kornia_aug.RandomGaussianBlur(
        (5, 5), (sigma_blur_min, sigma_blur_max), p=0.75, keepdim=True)

    # Gaussian noise
    gaussian_noise = kornia_aug.RandomGaussianNoise(
        mean=0.0, std=sigma_noise, p=0.75, keepdim=True)

    # Motion blur
    # direction_min, direction_max = -1.0, 1.0
    # motion_blur = kornia_aug.RandomMotionBlur((3, 3), (0, 360), direction=(direction_min, direction_max), p=0.75, keepdim=True)

    return torch.nn.Sequential(random_brightness, random_contrast, gaussian_blur, gaussian_noise)


# TODO GeometryAugsModule
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
