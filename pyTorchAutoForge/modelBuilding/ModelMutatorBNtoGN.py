import torch.nn as nn

class ModelMutatorBNtoGN():
    def __init__(self, model:nn.Module, numOfGroups: int = 32) -> None:
        self.numOfGroups = numOfGroups
        self.model = model

    def MutateBNtoGN(self) -> nn.Module:
        """Mutates all BatchNorm layers to GroupNorm layers in the model"""

        self.replace_batchnorm_with_groupnorm(self.model, self.numOfGroups)

        return self.model

    def replace_batchnorm_with_groupnorm(self, module, numOfGroups):
        """Recursively replaces BatchNorm2d layers with GroupNorm, adjusting num_groups dynamically."""
        for name, layer in module.named_children():
            if isinstance(layer, nn.BatchNorm2d):
                num_channels = layer.num_features
                # Check if divisible by num_groups
                if num_channels % self.numOfGroups != 0:
                    num_groups = self.find_divisible_groups(num_channels)
                else:
                    num_groups = self.numOfGroups
                # Replace BN with GroupNorm
                setattr(module, name, nn.GroupNorm(num_groups, num_channels))
            else:
                # Recurse to find all BatchNorm layers
                self.replace_batchnorm_with_groupnorm(layer, numOfGroups)

    def find_divisible_groups(self, num_channels):
        """Finds an appropriate number of groups for GroupNorm that divides num_channels."""
        # Start with 32, or reduce it if it doesn’t divide num_channels
        for groups in [16, 8, 4, 2]:  # Attempt standard group sizes
            if num_channels % groups == 0:
                return groups
        raise ValueError(f"Could not find a suitable number of groups for {num_channels} channels.")
