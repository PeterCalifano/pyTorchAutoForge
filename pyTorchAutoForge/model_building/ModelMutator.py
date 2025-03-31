"""
    This module provides a ModelMutator class for modifying PyTorch models.

    The ModelMutator class allows for in-place or external mutation of models by replacing
    specific layers (e.g., BatchNorm2d) with alternative layers (e.g., GroupNorm) based on
    the specified mutation type.

    :raises ValueError: If no mutation type is provided during mutation.
    :raises ValueError: If the mutation type is invalid or unsupported.
    :raises ValueError: If a suitable number of groups for GroupNorm cannot be determined.
    :return: The mutated PyTorch model.
    :rtype: nn.Module
"""
from torch import nn
from enum import Enum

class EnumMutations(Enum):
    BNtoGN = 0
    #GNtoBN = 1

# TODO upgrade mutator to allow replacements enumerated in EnumMutations
class ModelMutator():
    def __init__(self, model: nn.Module, numOfGroups: int = 32, in_place: bool = False, mutation_type: EnumMutations | None = EnumMutations.BNtoGN) -> None:
        self.numOfGroups = numOfGroups
        self.model = model
        self.mutation_type = mutation_type

        # Define the mapping between EnumMutations and methods
        self.mutation_methods = {
            EnumMutations.BNtoGN: self.replace_batchnorm_with_groupnorm_,
            # Add other mappings here
        }

        # Run mutation in-place if required (user can then get model directly)
        if in_place:
            self.mutate()

    def mutate(self, mutation_type: EnumMutations | None = None) -> nn.Module:
        """
        mutate _summary_ 

        _extended_summary_

        :return: _description_
        :rtype: nn.Module
        """        

        if mutation_type is None and self.mutation_type is None:
            raise ValueError("No mutation type provided. Please specify a mutation type.")
        elif mutation_type is None:
            mutation_type = self.mutation_type

        # Retrieve the corresponding method based on the mutation_type
        mutation_method = self.mutation_methods.get(mutation_type) if mutation_type is not None else None

        if mutation_method is not None:
            # Call the method with necessary arguments
            mutation_method(self.model, self.numOfGroups)
        else:
            raise ValueError(
                f"Mutation type {mutation_type} is invalid or currently not supported.")

        return self.model
    

    def replace_batchnorm_with_groupnorm_(self, module, numOfGroups):
        """
        Recursively replaces BatchNorm2d layers with GroupNorm, adjusting num_groups dynamically.

        :param module: The module to process.
        :type module: nn.Module
        :param numOfGroups: Desired number of groups for GroupNorm.
        :type numOfGroups: int
        """
        for name, layer in module.named_children():
            if isinstance(layer, nn.BatchNorm2d):
                num_channels = layer.num_features
                # Check if divisible by num_groups
                if num_channels % self.numOfGroups != 0:
                    num_groups = self.find_divisible_groups_(num_channels)
                else:
                    num_groups = self.numOfGroups
                # Replace BN with GroupNorm
                setattr(module, name, nn.GroupNorm(num_groups, num_channels))
            else:
                # Recurse to find all BatchNorm layers
                self.replace_batchnorm_with_groupnorm_(layer, numOfGroups)

    def find_divisible_groups_(self, num_channels):
        """Finds an appropriate number of groups for GroupNorm that divides num_channels."""
        # Start with 32, or reduce it if it doesn’t divide num_channels
        for groups in [16, 8, 4, 2]:  # Attempt standard group sizes
            if num_channels % groups == 0:
                return groups
        raise ValueError(f"Could not find a suitable number of groups for {num_channels} channels.")
