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
