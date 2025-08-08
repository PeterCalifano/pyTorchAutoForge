import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, is_tensor
import numpy as np

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

    def __init__(self, 
                 input_resolution: tuple[int, int], 
                 num_input_channels : int = 1,
                 downsampling_res_factor: int | float | tuple[float, float] | tuple[int, int] = (1.0, 1.0),
                 expectation_normalization_factor: int | float | tuple[float, float] | tuple[int, int] = (1.0, 1.0)) -> None:
        super().__init__()
        
        self.height, self.width = input_resolution
        self.num_input_channels = num_input_channels

        # Create coordinate buffers normalized to [-1, 1]
        #x_coords = torch.linspace(-1.0, 1.0, self.width)
        #y_coords = torch.linspace(-1.0, 1.0, self.height)

        # Pixel‐based coordinate buffers, shape (HW,)
        #xs = torch.arange(0, self.width, dtype=torch.float32)        
        #ys = torch.arange(0, self.height, dtype=torch.float32)

        #xs_flat = xs.repeat(self.height)            # [0,1,..,W-1, 0,1,..,W-1, …]
        #ys_flat = ys.repeat_interleave(self.width)  # [0..0,1..1,..H-1..H-1]

        xs_flat = torch.tensor(np.tile(np.arange(self.width), self.height),
                               dtype=torch.float32).unsqueeze(0)
        ys_flat = torch.tensor(np.repeat(np.arange(self.height), self.width),
                               dtype=torch.float32).unsqueeze(0)

        # If int or float, convert to tuple
        if isinstance(downsampling_res_factor, (int, float)):
            downsampling_res_factor = (float(downsampling_res_factor), float(downsampling_res_factor))

        elif isinstance(downsampling_res_factor, tuple) and len(downsampling_res_factor)== 2:
            downsampling_res_factor = (float(downsampling_res_factor[0]), float(downsampling_res_factor[1]))

        else:
            raise ValueError("downsampling_res_factor must be an int, float or a tuple of two floats.")

        if isinstance(expectation_normalization_factor, (int, float)):
            expectation_normalization_factor = (float(expectation_normalization_factor), float(expectation_normalization_factor))
        
        elif isinstance(expectation_normalization_factor, tuple) and len(expectation_normalization_factor) == 2:
            expectation_normalization_factor = (float(expectation_normalization_factor[0]), float(expectation_normalization_factor[1]))

        else:
            raise ValueError("expectation_normalization_factor must be an int, float or a tuple of two floats.")


        # Register data buffers
        self.register_buffer("x_coords", xs_flat.view(1,1,-1))  # shape (W,)
        self.register_buffer("y_coords", ys_flat.view(1,1,-1))  # shape (H,)
        self.register_buffer('downsampling_res_factor',
                             torch.tensor(downsampling_res_factor, dtype=torch.float32))
        self.register_buffer('expectation_normalization_factor',
                             torch.tensor(expectation_normalization_factor, dtype=torch.float32))


    def forward(self, feature_map: Tensor) -> Tensor:
        B, C, H, W = feature_map.shape

        assert H == self.height and W == self.width, (f"Input feature_map size ({H}, {W}) must match module initialization ({self.height}, {self.width})")

        # Apply softmax over spatial dimensions per channel
        probability_mask_flat = F.softmax(feature_map.view(B, C, -1), dim=2) # shape (B, C, HW)

        # Expected x coordinates: sum over y dimension then weighted by x_coords
        x_expectation = (probability_mask_flat * self.x_coords).sum(dim=2)  # shape (B, C)

        # Expected y coordinates: sum over x dimension then weighted by y_coords
        y_expectation = (probability_mask_flat * self.y_coords).sum(dim=2)  # shape (B, C)

        # Scale coordinates by downsampling factor to the original input resolution
        pred_x = x_expectation * self.downsampling_res_factor[0]
        pred_y = y_expectation * self.downsampling_res_factor[1]

        # Normalize coordinates by expectation normalization factor
        pred_x = pred_x / self.expectation_normalization_factor[0]
        pred_y = pred_y / self.expectation_normalization_factor[1]

        # Stack coordinates into (B, C, 2) shape
        xy_pred_coordinates = torch.stack((pred_x, pred_y), dim=2)

        # Flatten to (B, -1) stacking keypoints (x,y) for each channel
        return xy_pred_coordinates.view(B, C * 2)



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
