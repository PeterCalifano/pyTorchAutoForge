# TODO implement augmentations manager class, providing easy to use and flexible interface to consistently apply all augmentations in one call after config. Provides direct interface for ImagesAugmentations module and additional 1D vectors, as well as sequenced augmentations.

from email.mime import image
import kornia.augmentation as K
from collections.abc import Sequence
from typing import Literal, TypeAlias, Any
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import colorama
from pyTorchAutoForge.utils.conversion_utils import torch_to_numpy, numpy_to_torch

from pyTorchAutoForge.datasets.ImagesAugmentation import ImageAugmentationsHelper, AugmentationConfig

# %% Type aliases
ndArrayOrTensor: TypeAlias = np.ndarray | torch.Tensor

@dataclass
class AugsManagerConfig:
    images_helper_config: AugmentationConfig | None = None


class AugmentationsManager(torch.nn.Module):
    def __init__(self, augs_manager_config: AugsManagerConfig):
        """
        __init__ _summary_

        _extended_summary_

        :param augs_manager_config: _description_
        :type augs_manager_config: AugsManagerConfig
        """
        super().__init__()

        # Store configuration
        self.config = augs_manager_config
        
        # Build images augmentations helper if config provided
        self.images_augs_helper : ImageAugmentationsHelper | None = None

        if self.config.images_helper_config is not None:
            self.images_augs_helper = ImageAugmentationsHelper(self.config.images_helper_config)
            


    def plug_augs_helper(self, image_augs_helper: ImageAugmentationsHelper):
        """
        plug_augs_helper _summary_

        _extended_summary_

        :param image_augs_helper: _description_
        :type image_augs_helper: ImageAugmentationsHelper
        :raises TypeError: _description_
        """

        # Assert input is valid
        if not(isinstance(image_augs_helper, ImageAugmentationsHelper)):
            raise TypeError(f"Input image_augs_helper must be of type ImageAugmentationsHelper, got {type(image_augs_helper)}") 
        
        if self.images_augs_helper is not None:
            print(colorama.Fore.YELLOW +
                  "[Warning] Replacing existing ImageAugmentationsHelper in AugmentationsManager." + colorama.Style.RESET_ALL)
            
        # Replace module
        self.images_augs_helper = image_augs_helper

        # Replace configuration in manager config
        self.config.images_helper_config = image_augs_helper.augs_cfg



    
