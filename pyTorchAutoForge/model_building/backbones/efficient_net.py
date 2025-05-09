from torch import nn
from torchvision import models
from .base_backbones import EfficientNetConfig, FeatureExtractorFactory


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

        elif cfg.output_type == 'spill_features':
            # For spill_features, keep individual stages
            feature_extractor_modules = modules[0]

            # NOTE the construction of self.feature_extractor completely determines which output forward() will return
            self.feature_extractor = nn.ModuleList(
                list(feature_extractor_modules.children()))

            # Add last layer (adaptive pooling) from modules
            self.feature_extractor.append(modules[1])

            # Build average pooling layer
            self.feature_processor = nn.ModuleDict()

            if self.cfg.feature_tapping_output_res is not None \
                    and self.cfg.feature_tapping_channel_size is not None:

                for key, target_res in self.cfg.feature_tapping_output_res.items():

                    # Adaptive max pooling layer
                    pooled_res = (4*target_res[0], 4*target_res[1])
                    adaptive_max_pool = nn.AdaptiveMaxPool2d(pooled_res)

                    # Convolutional 2d extractor layer
                    max_pool_extractor_layer = nn.Conv2d(in_channels=self.cfg.feature_tapping_channel_size[key],
                                                         out_channels=1,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=1)

                    # Max pooling layer
                    max_pool_extractor_out_layer = nn.AdaptiveMaxPool2d(
                        output_size=(2*target_res[0],
                                     2*target_res[1]))

                    # Flattening layer
                    flattening_layer = nn.Flatten()

                    # Activation function for features
                    features_activation_in = nn.PReLU(num_parameters=1)
                    features_activation_out = nn.PReLU(num_parameters=1)

                    # Feature mixing layer
                    feature_weighting = nn.Linear(
                        in_features=4 * target_res[0] * target_res[1],
                        out_features=self.cfg.feature_tapping_channel_size[key] * target_res[0] * target_res[1])

                    self.feature_processor[str(key)] = nn.Sequential(
                        adaptive_max_pool,
                        max_pool_extractor_layer,
                        max_pool_extractor_out_layer,
                        flattening_layer,
                        features_activation_in,
                        feature_weighting,
                        features_activation_out,
                    )
            else:
                raise ValueError(
                    "feature_tapping_channel_size must not be None when feature_tapping_output_res is provided and output type is {cfg.output_type}.")

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

        if self.cfg.output_type not in ['last', 'spill_features']:
            raise ValueError(
                f"Invalid output_type: {self.cfg.output_type}. Must be 'last' or 'spill_features'.")

        # Pass through feature extractor
        for layer in self.feature_extractor:
            x = layer(x)

            if self.cfg.output_type == 'spill_features':
                features.append(x)

        # Process selected features with average pooling
        if self.cfg.output_type == 'spill_features' and self.cfg.feature_tapping_output_res is not None:
            for key, target_res in self.cfg.feature_tapping_output_res.items():
                if key in self.feature_processor:
                    features[int(key)] = self.feature_processor[key](
                        features[int(key)])

        # Handle output and optional head
        if self.cfg.output_type == 'last':
            out = x
            if self.output_layer is not None:
                out = self.output_layer(out.view(out.size(0), -1))
            return out

        elif self.cfg.output_type == 'spill_features':
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
