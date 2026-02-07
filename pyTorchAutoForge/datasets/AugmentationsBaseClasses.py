# TODO understand how to use for labels processing
from kornia.augmentation import AugmentationSequential
#import albumentations
import torch
from kornia import augmentation as kornia_aug
from torch import nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
# , cupy # cupy for general GPU acceleration use (without torch) to test
import numpy as np
from enum import Enum

# TODO 
class AugsBaseClass(nn.Module):
    """
    AugsBaseClass _summary_

    _extended_summary_

    :param nn: _description_
    :type nn: _type_
    """
    
    def __init__(self):
        super(AugsBaseClass, self).__init__()


# %% Base error models classes
class EnumComputeBackend(Enum):
    NUMPY = 1
    TORCH = 2
    CUPY = 3


class BaseErrorModel(nn.Module, ABC):
    """
    BaseErrorModel _summary_

    _extended_summary_

    :param nn: _description_
    :type nn: _type_
    :param ABC: _description_
    :type ABC: _type_
    """

    def __init__(self, shape: list, device: str = 'cpu') -> None:
        super(BaseErrorModel, self).__init__()
        # Default fidelity level for all models
        self.shape = shape  # Must be [B, C, H, W] for images
        self.device = device

        # Internal state
        self.last_sample = None

        # Assert input shape validity
        assert len(
            self.shape) >= 1, "Shape must be a list of at least 1 dimensions specifying the shape of the input tensor"

    @abstractmethod
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to apply error model to input tensor.
        Args:
            tensor (torch.Tensor): Input tensor to which the error model will be applied.
        Returns:
            torch.Tensor: Tensor with error model applied.
        """
        raise NotImplementedError(
            "This is an abstract method and must be implemented by the derived class")

    @abstractmethod
    def sample_realization(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to compute error realization for the specific error model.
        """
        raise NotImplementedError(
            "This is an abstract method and must be implemented by the derived class")

    def _assert_sample_validity(self, sample: torch.Tensor) -> None:
        """
        _assert_sample_validity _summary_

        _extended_summary_

        :param sample: _description_
        :type sample: torch.Tensor
        :raises ValueError: _description_
        """

        # Check sample shape validity
        if sample.shape != tuple(self.shape):
            raise ValueError(
                f"Sample shape {sample.shape} does not match expected shape {self.shape}.")

        # Validity checks
        assert sample is not None, "Error realization must not be None!"
        assert sample.shape == tuple(self.shape), f"Error sample shape {sample.shape} must match expected shape {self.shape} for additive error application."
        assert sample.device == torch.device(self.device), f"Error sample device {sample.device} must match expected device {self.device} for additive error application."
        assert sample.isnan().sum(
        ) == 0, "Error sample contains NaN values, cannot apply additive error."

class BaseGainErrorModel(BaseErrorModel):
    """
    BaseGainErrorModel _summary_

    _extended_summary_

    :param BaseErrorModel: _description_
    :type BaseErrorModel: _type_
    """

    def __init__(self, shape) -> None:
        super(BaseGainErrorModel, self).__init__(shape)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        # Sample error realization
        error_sample = self.sample_realization(X)
        self._assert_sample_validity(error_sample)

        # Determine operation based on size of error sample
        if error_sample.shape == X.shape:
            # Element-wise multiplication
            return X * error_sample
        
        elif error_sample.shape[0] == X.shape[0] and error_sample.shape[1] == X.shape[1]:
            # Batch-wise multiplication (broadcasting over spatial dimensions)
            return X * error_sample[:, :, None, None]
        
        elif error_sample.shape[0] == X.shape[0] and error_sample.shape[1] == 1:
        
            # Batch-wise matrix multiplication
            return torch.bmm(input=X, mat2=error_sample)
        
        else:
            raise ValueError(f"Error sample shape {error_sample.shape} is not compatible with input tensor shape {X.shape} for gain error application.")    
    
class BaseAddErrorModel(BaseErrorModel):
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, inputShape) -> None:
        super(BaseAddErrorModel, self).__init__(inputShape)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:

        # Sample error realization
        error_sample = self.sample_realization(tensor)
        self._assert_sample_validity(error_sample)

        # Apply additive error to input tensor
        return tensor + error_sample






