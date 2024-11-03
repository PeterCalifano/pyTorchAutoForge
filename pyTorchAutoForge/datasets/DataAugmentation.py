from kornia.augmentation import AugmentationSequential # DEVNOTE: TODO understand how to use for labels processing
import albumentations
from typing import Union
import torch
from kornia import augmentation as kornia_aug
from torch import nn


def build_kornia_augs(sigma_noise: float, sigma_blur: Union[tuple, float] = (0.0001, 1.0),
                      brightness_factor: Union[tuple, float] = (0.0001, 0.01),
                      contrast_factor: Union[tuple, float] = (0.0001, 0.01)) -> torch.nn.Sequential:

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


class ImagesAugsModule(nn.Module):
    def __init__(self, sigma_noise: float, sigma_blur: Union[tuple, float] = (0.0001, 1.0),
                    brightness_factor: Union[tuple, float] = (0.0001, 0.01),
                    contrast_factor: Union[tuple, float] = (0.0001, 0.01), unnormalize_before: bool = False):
        super(ImagesAugsModule, self).__init__()

        # Store augmentations data
        self.sigma_noise = sigma_noise
        self.sigma_blur = sigma_blur
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

        self.augmentations = build_kornia_augs(sigma_noise, sigma_blur, brightness_factor, contrast_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.unnormalize_before:
            x = 255.0 * x

        x = self.augmentations(x)

        if self.unnormalize_before:
            x = x / 255.0

        return x


class GeometryAugsModule(nn.Module):
    def __init__():
        pass

    def forward(self, x: torch.Tensor, labels: Union[torch.Tensor, tuple[torch.Tensor]]) -> torch.Tensor:
        # TODO define interface (input, output format and return type)
        x, labels = self.augmentations(x, labels)

        return x, labels


    