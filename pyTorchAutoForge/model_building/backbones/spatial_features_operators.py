import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def spatial_softargmax(feature_map):
    # feature_map shape: (N, C, H, W)
    N, C, H, W = feature_map.shape
    # Apply softmax over HxW per channel
    prob = F.softmax(feature_map.view(N, C, -1), dim=-1).view(N, C, H, W)
    # Create coordinate grids normalized to [-1, 1]
    x_coords = torch.linspace(-1, 1, W, device=feature_map.device)
    y_coords = torch.linspace(-1, 1, H, device=feature_map.device)
    # Compute expected coordinates
    # sum over rows (y) and cols (x) weighted by probabilities
    # prob.sum(dim=2) collapses y dimension -> shape (N,C,W) for x
    x_expectation = (prob.sum(dim=2) * x_coords).sum(dim=2)  # (N, C)
    y_expectation = (prob.sum(dim=3) * y_coords).sum(dim=2)  # (N, C)
    # coordinates in normalized [-1,1] space
    return x_expectation, y_expectation


class SpatialKptFeatureSoftmaxLocator(nn.Module):
    """
    A module that computes the spatial soft-argmax of a feature map.
    It takes a feature map of shape (B, C, H, W) and returns the expected x and y coordinates
    for each channel, normalized to [-1, 1]. The output shape is (B, C) for both x and y coordinates.
    The resolution of the grid is given by the input size.
    Args:
        height (int): The height of the input feature map.
        width (int): The width of the input feature map.
    """     
    def __init__(self, height: int, width: int):
        super().__init__()
        
        self.height = height
        self.width = width

        # Create coordinate buffers normalized to [-1, 1]
        x_coords = torch.linspace(-1.0, 1.0, width)
        y_coords = torch.linspace(-1.0, 1.0, height)

        self.register_buffer("x_coords", x_coords)  # shape (W,)
        self.register_buffer("y_coords", y_coords)  # shape (H,)

    def forward(self, feature_map: Tensor) -> Tensor:
        B, C, H, W = feature_map.shape

        assert H == self.height and W == self.width, (f"Input feature_map size ({H}, {W}) must match module initialization ({self.height}, {self.width})")

        # Apply softmax over spatial dimensions per channel
        probability_mask = F.softmax(feature_map.view(B, C, -1), dim=-1).view(B, C, H, W)

        # Expected x coordinates: sum over y dimension then weighted by x_coords
        x_expectation = (probability_mask.sum(dim=2) * self.x_coords).sum(dim=2)  # shape (B, C)

        # Expected y coordinates: sum over x dimension then weighted by y_coords
        y_expectation = (probability_mask.sum(dim=3) * self.y_coords).sum(dim=2)  # shape (B, C)

        # Stack coordinates into (B, N, 2) shape
        xy_expected_coordinates = torch.stack((x_expectation, y_expectation), dim=-1)  # shape (B, C, 2)
        return xy_expected_coordinates


# Runnable example
if __name__ == "__main__":
    # Create a random feature map of shape (batch=2, channels=4, height=5, width=7)
    feature_map = torch.randn(2, 4, 5, 7)
    module = SpatialKptFeatureSoftmaxLocator(5, 7)
    xy_out = module(feature_map)

    print("Output shapes:")
    print("xy:", xy_out.shape)  # expected (2, 4, 2)
    print("\nSample outputs:")
    print("xy:", xy_out)

    # Test ONNX export compatibility
    dummy_input = torch.randn(1, 3, 5, 7)
    
    try:
        torch.onnx.export(
            module,
            dummy_input,
            "spatial_softargmax.onnx",
            opset_version=11,
            input_names=["input"],
            output_names=["xy_out"]
        )
        print("\nONNX export succeeded: 'spatial_softargmax.onnx'")
    except Exception as e:
        print(f"\nONNX export failed: {e}")
