# TODO
import albumentations
from typing import Union
import torch
from kornia import augmentation as kornia_aug

def build_kornia_augs(sigma_noise: float, sigma_blur: Union[tuple, float] = (0.0001, 1.0),
                      brightness_factor: Union[tuple, float] = (0.0001, 0.01),
                      contrast_factor: Union[tuple, float] = (0.0001, 0.01)) -> torch.nn.Sequential:

    # Define kornia augmentation pipeline

    # Gaussian noise
    gaussian_noise = kornia_aug.RandomGaussianNoise(
        mean=0.0, std=sigma_noise, p=0.75, keepdim=True)

    # Colour jiggle DEVNOTE: has issues. It returns a completely black image
    brightness_min, brightness_max = brightness_factor if isinstance(
        brightness_factor, tuple) else (brightness_factor, brightness_factor)
    contrast_min, contrast_max = contrast_factor if isinstance(
        contrast_factor, tuple) else (contrast_factor, contrast_factor)

    # colour_jiggle = kornia_aug.ColorJiggle(brightness=(brightness_min, brightness_max),
    #                                    contrast=(contrast_min, contrast_max), p=0.75, keepdim=True)

    # Gaussian Blur
    sigma_blur_min, sigma_blur_max = sigma_blur if isinstance(
        sigma_blur, tuple) else (sigma_blur, sigma_blur)
    gaussian_blur = kornia_aug.RandomGaussianBlur(
        (5, 5), (sigma_blur_min, sigma_blur_max), p=0.75, keepdim=True)

    # Motion blur
    direction_min, direction_max = -1.0, 1.0
    motion_blur = kornia_aug.RandomMotionBlur((3, 3), (0, 360), direction=(
        direction_min, direction_max), p=0.75, keepdim=True)

    return torch.nn.Sequential(gaussian_noise, gaussian_blur)
