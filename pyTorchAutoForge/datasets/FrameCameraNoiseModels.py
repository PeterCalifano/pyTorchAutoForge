# TODO understand how to use for labels processing
from kornia.augmentation import AugmentationSequential
import albumentations
import torch
from kornia import augmentation as kornia_aug
from torch import nn
from abc import ABC, abstractmethod
import pytest  # For unit tests
from dataclasses import dataclass
#import cupy # cupy for general GPU acceleration use (without torch) to test
import numpy as np
from enum import Enum

from pyTorchAutoForge.datasets.DataAugmentation import EnumComputeBackend


# %% Error models classes implementations
def SamplePoissonRV(rates: torch.Tensor | np.ndarray, float, inputShape: list | None = None,
                    enumComputeBackend: EnumComputeBackend = EnumComputeBackend.TORCH):

    # DEVNOTE: rates determines shape if inputShape is None
    if enumComputeBackend == EnumComputeBackend.TORCH:
        # if inputShape is not None and rates is scalar --> rates = torch.ones(inputShape) * rates

        # Generate Poisson random variables
        return torch.poisson(rates)

    elif enumComputeBackend == EnumComputeBackend.NUMPY:
        raise NotImplementedError("Compute operation not implemented")
    elif enumComputeBackend == EnumComputeBackend.CUPY:
        raise NotImplementedError("Compute operation not implemented")
    else:
        raise ValueError("Invalid compute backend")


class ShotNoiseModel(BaseAddErrorModel):
    """
    Shot noise model for CMOS image sensors. The model is based on Poisson random variables.
    """

    def __init__(self, inputShape, shotNoiseRates) -> None:
        super(ShotNoiseModel, self).__init__(inputShape)
        self.shotNoiseRates = shotNoiseRates

    def realizeError(self, X: torch.Tensor) -> None:
        """Compute realization of shot noise error model using Poisson random variables.

        Returns:
            None: None
        """
        self.errorRealization = SamplePoissonRV(
            self.shotNoiseRates, self.inputShape).to(self.device)

        raise NotImplementedError("Compute operation not implemented")


class ResponseNonUniformityModel(BaseGainErrorModel):
    """_summary_

    Args:
        BaseGainErrorModel (_type_): _description_
    """

    def __init__(self, inputShape) -> None:
        super(ResponseNonUniformityModel, self).__init__(inputShape)

    def realizeError(self, X: torch.Tensor) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
        raise NotImplementedError("Compute operation not implemented")


class PixelCrossTalkModel(BaseErrorModel):
    """_summary_

    Args:
        BaseErrorModel (_type_): _description_
    """

    def __init__(self, inputShape) -> None:
        super(PixelCrossTalkModel, self).__init__(inputShape)

    def realizeError(self, X: torch.Tensor) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
        raise NotImplementedError("Compute operation not implemented")


class DarkCurrentModel(BaseAddErrorModel):
    """_summary_

    Args:
        BaseAddErrorModel (_type_): _description_
    """

    def __init__(self, inputShape) -> None:
        super(DarkCurrentModel, self).__init__(inputShape)
        # No DCNU model by default
        self.DarkCurrentResponseNonUniformity: ResponseNonUniformityModel = None
        # TODO: need to implement a way to compute the realization at forward time

    def realizeError(self, X: torch.Tensor) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
        # Compute realization of DCNU if any
        if self.DarkCurrentResponseNonUniformity is not None:
            self.DarkCurrentResponseNonUniformity.realizeError()

        # Compute realization of dark current
        self.errorRealization  # TODO: implement realization computation

        # Combine DCNU and dark current realization and store in errorRealization
        self.errorRealization = self.DarkCurrentResponseNonUniformity.forward(
            self.errorRealization)  # DCNU is a gain error model


class RowReadoutNoiseModel(BaseAddErrorModel):
    """_summary_

    Args:
        BaseAddErrorModel (_type_): _description_
    """

    def __init__(self, inputShape) -> None:
        super(RowReadoutNoiseModel, self).__init__(inputShape)

    def realizeError(self, X: torch.Tensor) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
        raise NotImplementedError("Compute operation not implemented")


# %% Combined error models
class ReadoutNoiseModel(BaseAddErrorModel):
    """Readout noise error model. The models employed by this class depends on the requested level of fidelity.
    - Low fidelity level only adds Gaussian noise to the input. 
    - Medium fidelity models Thermal, Row, and Column noise.
    - High fidelity includes all the above, Flicker (1/f) and RTS, Temporal Column noise, Vertical fixed pattern component.

    Args:
        BaseAddErrorModel (_type_): _description_
    """

    def __init__(self, inputShape: list,
                 noiseMean: float | torch.Tensor | tuple | np.ndarray | None = None,
                 noiseStd: float | tuple | np.ndarray | None = None) -> None:

        super(ReadoutNoiseModel, self).__init__(inputShape)
        self.noiseMean = noiseMean
        self.noiseStd = noiseStd
        # TODO list or dict of models to apply for medium and high fidelity levels
        self.errorModels = None

    def realizeError(self, X: torch.Tensor) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.enumModelFidelity == EnumModelFidelity.LOW:
            self.realizedError_low()
            # Add Gaussian noise
            # For ranges (std and mean not equal for all pixels):
            # torch.normal(mean=torch.arange(1., 11.),
            #             std=torch.arange(1, 0, -0.1))
        elif self.enumModelFidelity == EnumModelFidelity.MEDIUM:
            raise NotImplementedError("Compute operation not implemented")
            self.realizedError_medium()
        elif self.enumModelFidelity == EnumModelFidelity.HIGH:
            raise NotImplementedError("Compute operation not implemented")
            self.realizedError_high()
        else:
            raise ValueError(
                "Invalid fidelity level for ReadoutNoiseModel")

    def realizedError_low(self):
        """
        Implementation of LOW fidelity error realization
        """
        assert self.noiseMean is not None, "Mean value for Gaussian noise must be provided"
        assert self.noiseStd is not None, "Standard deviation value for Gaussian noise must be provided"

        self.errorRealization = torch.normal(
            mean=self.noiseMean, std=self.noiseStd, size=self.inputShape, device=self.device)


# %% TODO classes
@dataclass
class ErrorModelConfig():
    # Default values
    enumComputeBackend: EnumComputeBackend = EnumComputeBackend.TORCH


@dataclass
class CameraDetectorErrorModelConfig(ErrorModelConfig):
    pass


class CameraDetectorErrorsModel(AugsBaseClass):

    def __init__(self) -> None:
        super(CameraDetectorErrorsModel, self).__init__()
        self.errorModelsList = nn.ModuleDict()

        # TODO build error models from config, from input dict or list of error models

    def forward(self, imageAsDN: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:

        # Loop through error models
        for errorModel in self.errorModelsList:
            # Call forward method of each error model
            imageAsDN = errorModel(imageAsDN)

        return imageAsDN
    
# TODO ImagesAugsModule
class ImagesAugsModule(AugsBaseClass):
    def __init__(self, sigma_noise: float, sigma_blur: tuple | float = (0.0001, 1.0),
                 brightness_factor: tuple | float = (0.0001, 0.01),
                 contrast_factor: tuple | float = (0.0001, 0.01), unnormalize_before: bool = False):
        super(ImagesAugsModule, self).__init__()

        # Store augmentations data
        self.sigma_noise = sigma_noise
        self.sigma_blur = sigma_blur
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.unnormalize_before = unnormalize_before

        self.augmentations = build_kornia_augs(
            sigma_noise, sigma_blur, brightness_factor, contrast_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.unnormalize_before:
            x = 255.0 * x

        x = self.augmentations(x)

        if self.unnormalize_before:
            x = x / 255.0

        return x



# %% Module unit tests
def test_CameraDetectorErrorModel():

    # Test CameraDetectorErrorModel class
    errorModel = CameraDetectorErrorsModel()
    print(errorModel)


if __name__ == "__main__":
    test_CameraDetectorErrorModel()


