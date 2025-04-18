from typing import Literal
from pyTorchAutoForge.utils import GetDeviceMulti
from pyTorchAutoForge.setup import BaseConfigClass
from functools import singledispatch
from .efficient_net import EfficientNetBackbone, EfficientNetConfig
from torch import nn
from dataclasses import dataclass, field
from .input_adapters import BaseAdapterConfig, InputAdapterFactory


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
    pretrained: bool = True
    # Dimension of the final linear layer (if you want to add a linear layer)
    output_size: int | None = None
    remove_classifier: bool = True
    device: str = GetDeviceMulti()
    input_channels: int = 3 # Placeholder value

    # Whether to return only the final feature map, or all intermediate outputs
    output_type: Literal['last', 'features'] = 'last'


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

    def __post_init__(self):
        self.input_channels = 3


@FeatureExtractorFactory.register
def _(model_cfg: EfficientNetConfig) -> EfficientNetBackbone:
    return EfficientNetBackbone(model_cfg)

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
