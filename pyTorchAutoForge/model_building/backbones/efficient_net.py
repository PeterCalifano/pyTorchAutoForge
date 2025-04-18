from torch import nn
from torchvision import models
from .base_backbones import EfficientNetConfig

class EfficientNetBackbone(nn.Module):
    def __init__(self, cfg : EfficientNetConfig ):
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
            # Wrap as a single ModuleList so forward is simple
            self.feature_extractor = nn.ModuleList([nn.Sequential(*modules)])

        else:  # 'features'
            # e.g. each block as its own layer to capture intermediate outputs
            # assumes first child is the feature sequential, then splits
            feat_seq = modules[0]
            self.feature_extractor = nn.ModuleList(list(feat_seq.children()))

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
        features = []

        # Pass through feature extractor
        for layer in self.feature_extractor:
            x = layer(x)
            if self.cfg.output_type == 'features':
                features.append(x)

        # Handle output and optional head
        if self.cfg.output_type == 'last':
            out = x
            if self.output_layer is not None:
                out = self.output_layer(out.view(out.size(0), -1))
        else:
            # Append head output to the features list and return the whole list
            if self.output_layer is not None and features:
                last_feat = features[-1]
                head_out = self.output_layer(last_feat.view(last_feat.size(0), -1))
                features.append(head_out)
            out = features

        return out
