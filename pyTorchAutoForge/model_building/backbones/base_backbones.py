from typing import Literal
from pyTorchAutoForge.utils import GetDeviceMulti
from pyTorchAutoForge.setup import BaseConfigClass
from functools import singledispatch
from torch import nn
from dataclasses import dataclass, field
from pyTorchAutoForge.model_building.backbones.input_adapters import BaseAdapterConfig, InputAdapterFactory
import torch
import numpy as np  

@dataclass
class FeatureExtractorConfig(BaseConfigClass):
    """
    Configuration for a feature extractor backbone.

    Attributes:
        model_name: name of the backbone model (e.g. 'efficientnet_b0').
        input_resolution: input resolution for the model.
        pretrained: whether to use pretrained weights.
        num_classes: number of classes for the classification head.
    """
    adapter_config : BaseAdapterConfig | None = None
    input_resolution: tuple[int, int] = (512, 512)
    pretrained: bool = False
    # Dimension of the final linear layer (if you want to add a linear layer)
    output_size: int | None = None
    remove_classifier: bool = True
    device: torch.device | str | None = None
    input_channels: int = 3 # Placeholder value

    # Whether to return only the final feature map, or all intermediate outputs
    output_type: Literal['last', 'spill_features', 'spatial_features'] = 'last'

    def __post_init__(self):
        if self.device is None:
            self.device = GetDeviceMulti()

@dataclass
class BackboneConfig(BaseConfigClass):
    """
    Configuration for a backbone model.

    Attributes:
        adapter_cfg: configuration for the input adapter (if any).
        backbone_cfg: configuration for the backbone model.
    """
    feature_extractor_cfg: FeatureExtractorConfig 
    adapter_cfg: BaseAdapterConfig | None = None

    def __post_init__(self):
        # Validate adapter and extractor configurations
        if self.feature_extractor_cfg is None:
            raise ValueError(" config must be provided.")
        
        # Check if adapter configuration matches the feature extractor input
        if self.adapter_cfg is not None:
            if self.adapter_cfg.channel_sizes[-1] != self.feature_extractor_cfg.input_channels:
                raise ValueError("Adapter output channels must match feature extractor input channels.")
            
            # Check if output size matches the feature extractor input resolution
            if self.adapter_cfg.output_size[0] != self.feature_extractor_cfg.input_resolution[0] or \
               self.adapter_cfg.output_size[1] != self.feature_extractor_cfg.input_resolution[1]:
                raise ValueError("Adapter output size must match feature extractor input resolution.")
            

# Define factory with dispatch
@singledispatch
def FeatureExtractorFactory(model_cfg) -> nn.Module:
    """
    Build and return a backbone based on the provided config instance.
    New config types can be registered decorated with @FeatureExtractorFactory.register.
    """
    raise ValueError(f"No backbone registered for config type {type(model_cfg).__name__}")


def BackboneFactory(cfg: BackboneConfig) -> nn.Module:
    """
    Build full model pipeline: optional adapter followed by backbone.

    Args:
      cfg: BackboneConfig with optional adapter_cfg and feature_extractor_cfg.
    Returns:
      nn.Sequential stacking adapter (if any) then backbone.
    """
    modules = []
    if cfg.adapter_cfg is not None:
        modules.append(InputAdapterFactory(cfg.adapter_cfg))

    modules.append(FeatureExtractorFactory(cfg.feature_extractor_cfg))
    return nn.Sequential(*modules)

    
# %% Register dispatched functions for each backbone type
### EfficientNet
@dataclass
class EfficientNetConfig(FeatureExtractorConfig):
    # Which EfficientNet variant to use
    model_name: Literal['b0', 'b1', 'b2', 'b3', 'b4', 'b6'] = 'b0'
    feature_tapping_output_resolution_channels: dict[str, dict[str, tuple[int, int] | int]] | None = field(default_factory=lambda: {"1": {"resolution": (32,32), "channels": 1, "linear_output_size": 128},})
    expectation_normalization_factor: int | float | tuple[float, float] = (1.0, 1.0)

    def __post_init__(self):
        self.input_channels = 3
        # Define output channel sizes for each EfficientNet variant
        if self.model_name == 'b0':
            feature_tapping_channel_out = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        else:
            raise ValueError(f"Channel sizes output (for each module) not defined for {self.model_name} variant. Please add it manually.")

        self.feature_tapping_channel_input_size: dict[str, int] | None = None

        # Build mapping for EfficientNet channel input sizes 
        if self.feature_tapping_output_resolution_channels is not None:

            # Define output channels for each feature_tapping key
            self.feature_tapping_channel_input_size = {key: 0 for key in self.feature_tapping_output_resolution_channels.keys()}

            # Define feature_tapping_channel_input_size for each stage
            # TODO (PC) clarify what is the scope of this loop?
            for key, value in self.feature_tapping_output_resolution_channels.items():

                # Handle value being a tuple of length 2 or 3
                if isinstance(value, dict):
                    
                    if len(value) == 2:
                        key_resolution, key_channels = value.keys()

                        # Get resolution and channels from the config
                        resolution = self.feature_tapping_output_resolution_channels[key][key_resolution]
                        channels = self.feature_tapping_output_resolution_channels[key][key_channels]

                        if self.output_type == 'spill_features':
                            linear_output_size = 128
                            print(f"\033[38;5;208mWARNING: missing linear_output_size for spill_features variant. Default value of {linear_output_size} will be used.\033[0m")
                            
                    elif len(value) == 3:
                        key_resolution, key_channels, key_linear_output_size = value.keys()

                        # Get resolution and channels from the config
                        resolution = self.feature_tapping_output_resolution_channels[key][key_resolution]
                        channels = self.feature_tapping_output_resolution_channels[key][key_channels]
                        linear_output_size = self.feature_tapping_output_resolution_channels[key][key_linear_output_size]

                        # Check validity of linear_output_size
                        assert isinstance(linear_output_size, int), "Linear output size must be a scalar integer."
                        assert linear_output_size > 0, "Linear output size must be a positive integer."

                        if self.output_type == 'spatial_features':
                            print(f"\033[38;5;208mWARNING: linear_output_size provided but not used for spatial features. It will be ignored.\033[0m")
                    else:
                        raise ValueError(f"Value for key {key} must be a tuple of length 2 or 3.")
                    
                    # Check validity of settings
                    assert isinstance(resolution, (tuple, list)) and len(resolution) == 2, "Resolution must be a tuple or list of two integers."
                    assert np.all(np.array(resolution) >
                                    0), "Resolution must be a positive integer."

                    assert isinstance(channels, int), "Channels must be a scalar integer."
                    assert channels > 0, "Channels must be a positive integer."

                else:
                    raise ValueError(f"Value for key {key} must be a dict.")

                # Check key is a number
                if not key.isdigit():
                    raise ValueError(f"Key {key} is not a valid number. Must be the index of the module from which the features are spilled.")

                if int(key) >= len(feature_tapping_channel_out):
                    raise ValueError(f"Key {key} exceeds the number of EfficientNet children modules.")
                
                self.feature_tapping_channel_input_size[key] = feature_tapping_channel_out[int(key)]

        else:
            # Build dict for all feature tapping channels (input sizes)
            self.feature_tapping_channel_input_size = {str(
                i): feature_tapping_channel_out[i] for i in range(len(feature_tapping_channel_out))}

### ResNet
@dataclass
class ResNetConfig(FeatureExtractorConfig):
    # Which ResNet variant to use
    model_name: Literal['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] = 'resnet18'

    def __post_init__(self):
        self.input_channels = 3

#@FeatureExtractorFactory.register
#def _(model_cfg: ResNetConfig):
#    return ResNetBackbone(model_cfg)
######################
