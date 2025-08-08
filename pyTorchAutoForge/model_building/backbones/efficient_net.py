from torch import nn
from torchvision import models
from .base_backbones import EfficientNetConfig, FeatureExtractorFactory

import torch
import torch.nn.functional as F
from pyTorchAutoForge.model_building.backbones.spatial_features_operators import SpatialKptFeatureSoftmaxLocator


class EfficientNetBackbone(nn.Module):
    def __init__(self, cfg: EfficientNetConfig):
        super(EfficientNetBackbone, self).__init__()
        # Store configuration
        self.cfg = cfg

        # Dynamically pick the constructor, e.g. efficientnet_b0, b1, …
        effnet_factory = getattr(models, f'efficientnet_{cfg.model_name}')

        # Load with pretrained weights
        model = effnet_factory(weights=cfg.pretrained).to(cfg.device)

        # Extract the “features” part all children except the final classifier/sequential
        modules = list(model.children())[:-1]
        if cfg.output_type == 'last':
            # Wrap as a single ModuleList so that forward is simple
            self.feature_extractor = nn.ModuleList([nn.Sequential(*modules)])

        elif cfg.output_type == 'spill_features' or cfg.output_type == 'spatial_features':

            # For spill_features, keep individual stages
            feature_extractor_modules = modules[0]

            # NOTE the construction of self.feature_extractor completely determines which output forward() will return
            self.feature_extractor = nn.ModuleList(
                list(feature_extractor_modules.children()))

            # Add last layer (global adaptive pooling) from modules
            self.feature_extractor.append(modules[1])

            # Build average pooling layer
            self.feature_spill_preprocessor = nn.ModuleDict()

            if self.cfg.feature_tapping_output_resolution_channels is not None \
                    and self.cfg.feature_tapping_channel_input_size is not None:

                for key, value in self.cfg.feature_tapping_output_resolution_channels.items():

                    if len(value) == 2:
                        # For spatial features, expects two keys only
                        target_res_key, target_channels_key = value.keys()

                    elif len(value) == 3:
                        # For spill features, expects three keys
                        target_res_key, target_channels_key, target_linear_output_size = value.keys()

                    # Get target resolution and channels from configuration
                    target_res = self.cfg.feature_tapping_output_resolution_channels[
                        key][target_res_key]
                    target_channels = self.cfg.feature_tapping_output_resolution_channels[
                        key][target_channels_key]

                    # Adaptive max pooling layer
                    # Pool to 4 times the target resolution
                    #pooled_res = (4*target_res[0], 4*target_res[1])
                    #adaptive_max_pool = nn.AdaptiveMaxPool2d(pooled_res)

                    # Convolutional 2d extractor layer
                    max_pool_extractor_layer = nn.Conv2d(in_channels=self.cfg.feature_tapping_channel_input_size[key],
                                                         out_channels=target_channels,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=1)

                    # Max pooling layer
                    max_pool_extractor_out_layer = nn.AdaptiveMaxPool2d(
                        output_size=(target_res[0], target_res[1]))

                    # Activation function for output features maps
                    features_spill_activations = nn.PReLU(
                        num_parameters=target_channels)

                    if cfg.output_type == 'spill_features':  # Intensity-like feature output type
                        if 'linear_output_size' not in cfg.feature_tapping_output_resolution_channels[key].keys():
                            raise ValueError(
                                "feature_tapping_output_resolution_channels must contain 'linear_output_size' for intensity-like output type (spill_features mode).")

                        # Build flattening and linear layer to match expected output size
                        self.feature_spill_preprocessor[str(key)] = nn.Sequential(
                            max_pool_extractor_layer,
                            max_pool_extractor_out_layer,
                            features_spill_activations,
                            nn.Flatten(),
                            nn.Linear(in_features=target_channels * target_res[0] * target_res[1],
                                      out_features=cfg.feature_tapping_output_resolution_channels[key][target_linear_output_size])
                        )

                    else:  # Spatial features output type

                        if target_res[0] > cfg.input_resolution[0] or target_res[1] > cfg.input_resolution[1]:
                            raise ValueError(
                                f"Target resolution {target_res} must not exceed input resolution {cfg.input_resolution} at stage {key}.")
                        
                        # Compute downsampling factor from original image resolution to target resolution
                        downsampling_res_factor_ = (cfg.input_resolution[0] / target_res[0],
                                                    cfg.input_resolution[1] / target_res[1])

                        # Build softargmax layer to extract spatial features for each channel
                        # Outputs will be of shape (B, C, 2) where 2 is for x and y coordinates, C channels
                        spatial_kpt_extractor = SpatialKptFeatureSoftmaxLocator(input_resolution=target_res,
                                                                                num_input_channels=target_channels,
                                                                                downsampling_res_factor=downsampling_res_factor_,
                                                                                expectation_normalization_factor=cfg.expectation_normalization_factor)

                        self.feature_spill_preprocessor[str(key)] = nn.Sequential(max_pool_extractor_layer, 
                                                                                  max_pool_extractor_out_layer,
                                                                                  features_spill_activations,
                                                                                  spatial_kpt_extractor
                                                                                  )

            else:
                raise ValueError(
                    "feature_tapping_channel_input_size must not be None when feature_tapping_output_resolution_channels is provided and output type is {cfg.output_type}.")

        else:
            raise ValueError(
                f"Invalid output_type: {cfg.output_type}. Must be 'last' or 'spill_features'.")

        # Additional final layer as adapter
        if cfg.output_size is not None:
            # Get number of channels from last conv output by using known mapping for EfficientNet variants.
            final_ch = model.classifier[1].in_features
            self.output_layer: nn.Linear | None = nn.Linear(
                final_ch, cfg.output_size).to(cfg.device)
        else:
            self.output_layer = None

        if self.output_layer is not None:
            self.output_layer.to(cfg.device)

    def forward(self, x):
        features = []  # TODO this is the lazy way. It would be much faster by preallocating the list size, one way could be to run inference once and store the sizes, since these will not change at runtime.

        if self.cfg.output_type not in ['last', 'spill_features', 'spatial_features']:
            raise ValueError(
                f"Invalid output_type: {self.cfg.output_type}. Must be 'last', 'spill_features', or 'spatial_features'.")
        
        # Pass through feature extractor
        for layer in self.feature_extractor:
            x = layer(x)

            if self.cfg.output_type == 'spill_features' or self.cfg.output_type == 'spatial_features':
                features.append(x)

        # Process selected features with average pooling
        if (self.cfg.output_type == 'spill_features' or self.cfg.output_type == 'spatial_features') and self.cfg.feature_tapping_output_resolution_channels is not None:

            for key in self.cfg.feature_tapping_output_resolution_channels.keys():

                if key in self.feature_spill_preprocessor:
                    features[int(key)] = self.feature_spill_preprocessor[key](
                        features[int(key)])

        # Handle output and optional head
        if self.cfg.output_type == 'last':
            out = x
            if self.output_layer is not None:
                out = self.output_layer(out.view(out.size(0), -1))
            return out

        elif self.cfg.output_type == 'spill_features' or self.cfg.output_type == 'spatial_features':

            # Append head output to the features list and return the whole list
            if self.output_layer is not None and len(features) > 0:
                last_feat = features[-1]
                head_out = self.output_layer(
                    last_feat.view(last_feat.size(0), -1))
                features.append(head_out)

            out = features
            return out
    


# Define factory method for EfficientNet backbone
@FeatureExtractorFactory.register
def _(model_cfg: EfficientNetConfig) -> EfficientNetBackbone:
    return EfficientNetBackbone(model_cfg)
