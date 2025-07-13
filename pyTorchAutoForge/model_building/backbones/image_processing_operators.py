import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from pyTorchAutoForge.utils import torch_to_numpy, numpy_to_torch
# Optional differentiable operations via Kornia
try:
    import kornia.filters as KF
    import kornia
    USE_KORNIA = True
except ImportError:
    USE_KORNIA = False

# Auxiliary functions for handling different formats
def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure input is 4D batch tensor (B, C, H, W).
    Accepts 2D (H, W), 3D (C, H, W), or 4D (B, C, H, W).
    """
    if x.dim() == 2:
        # Interpret as (H, W), thus unsqueeze batch and channel dims
        return x.unsqueeze(0).unsqueeze(0)
    
    if x.dim() == 3:
        # Interpret as (C, H, W), thus unsqueeze batch dim
        return x.unsqueeze(0)
    
    if x.dim() == 4:
        # Assumed format (B, C, H, W)
        return x
    
    raise ValueError(f"Unsupported tensor dims {x.dim()}, expected 2,3, or 4.")


def _restore_shape(out: torch.Tensor, orig_dim: int) -> torch.Tensor:
    """
    Restore output shape to match original input dims.
    """
    # Input is assumed to be 4D (B, C, H, W) after processing

    if orig_dim == 4:
        return out  # Return (B, C, H, W)

    if orig_dim == 3:
        return out[0]  # Return (C, H, W)
    
    if orig_dim == 2:
        return out[0,0] # Return (H, W)

    raise ValueError(f"Unsupported original dimension {orig_dim}, expected 2, 3, or 4.")


def ConvertToLuminance(x: torch.Tensor) -> torch.Tensor:

    C = x.dim()
    x_bchw = _ensure_batch(x)

    # If C > 1, convert to luminance if C = 3 or mean over channels
    if C > 1:
        if C == 3:
            r, g, b = x_bchw[:, 0:1], x_bchw[:, 1:2], x_bchw[:, 2:3]
            x_bchw = (0.299 * r + 0.587 * g + 0.114 * b)
        else:
            # Average across channels for other C
            x_bchw = x_bchw.mean(dim=1, keepdim=True)

    return x_bchw


class QuantileThresholdMask(nn.Module):
    """
    Compute a binary mask of pixels above threshold.
    Supports inputs of shape: (H, W), (C, H, W), (B, C, H, W).
    """

    def __init__(self, abs_thr: float | None = None, 
                 quantile: float = 0.85,
                 return_type: Literal['mask', 'masked_image'] = 'mask'):
        """
        Args:
            abs_thr: Absolute threshold value. If None, use quantile.
            quantile: Quantile value for thresholding (0 < quantile < 1).
            return_type: 'mask' for binary mask, 'masked_image' for masked image.
        """
        super().__init__()
        self.abs_thr = abs_thr
        self.quantile = quantile
        self.return_type: Literal['mask', 'masked_image'] = return_type

        # Check quantile value
        if not (0 <= quantile <= 1):
            raise ValueError(
                f"Quantile must be in (0, 1), got {quantile}.")
        
        # Check return type
        if return_type not in ['mask', 'masked_image']:
            raise ValueError(
                f"Invalid return type: {return_type}. Must be 'mask' or 'masked_image'.")
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:

        # Get dimensions and unsqueeze if needed
        orig_dim = image.dim()
        x_bchw = _ensure_batch(image)

        B, C, H, W = x_bchw.shape
        x_bchw = ConvertToLuminance(x_bchw)

        # Renew shape sizes
        B, C, H, W = x_bchw.shape

        if self.abs_thr is None:

            # Reshape to (B, H*W) for quantile calculation
            flat = x_bchw.view(B, -1)

            flat_masked = flat.masked_fill(
                flat == 0, float('nan'))    # zeros → NaN

            # Compute quantile ignoring NaNs
            thr_per_image = torch.nanquantile(
                flat_masked,
                self.quantile,
                dim=1,
                keepdim=True,
                interpolation='linear'
            ) 

            # Replace all NaN with 0.0
            thr_per_image = torch.nan_to_num(thr_per_image, nan=0.0)
            
            thr_per_image = thr_per_image.view(B, 1, 1, 1)
            mask = (x_bchw >= thr_per_image).float()

        else:
            # Apply absolute threshold
            mask = (x_bchw >= self.abs_thr).float()
        
        # Return 
        
        if self.return_type == 'masked_image':
            # If return_type is 'masked_image', apply mask to input
            masked_image = x_bchw * mask
            return _restore_shape(masked_image, orig_dim)
        
        elif self.return_type == 'mask':
            # Return binary mask
            return _restore_shape(mask, orig_dim)
        else:
            raise ValueError(
                f"Invalid return type: {self.return_type}. Must be 'mask' or 'masked_image'.")


class SobelGradient(nn.Module):
    """
    Compute Sobel gradient magnitude feature map.
    Differentiable via Kornia if available.
    """

    def __init__(self):
        super().__init__()
        if not USE_KORNIA:
            # Register custom kernels if Kornia is unavailable
            kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
            ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
            self.register_buffer('kx', kx.view(1, 1, 3, 3))
            self.register_buffer('ky', ky.view(1, 1, 3, 3))
        else:
            self.kornia_sobel = KF.Sobel(eps=1E-12)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unsqueeze input to (B, C, H, W)
        orig_dim = x.dim()
        x_bchw = _ensure_batch(x)

        B, C, H, W = x_bchw.shape
        x_bchw = ConvertToLuminance(x_bchw)

        # Renew shape sizes
        B, C, H, W = x_bchw.shape
        
        if USE_KORNIA:
            # Compute luminance image
            mag = self.kornia_sobel(x_bchw)

        else:
            device = x.device
            kx, ky = self.kx.to(device), self.ky.to(device)
            gx = F.conv2d(x_bchw, kx, padding=1) # type: ignore
            gy = F.conv2d(x_bchw, ky, padding=1)  # type: ignore

            # Compute gradient magnitude feature map
            mag = torch.sqrt(gx * gx + gy * gy)

        out = mag
        return _restore_shape(out, orig_dim)


class LaplacianOfGaussian(nn.Module):
    """
    Compute Laplacian of Gaussian feature map.
    Differentiable via Kornia if available.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()

        self.kernel_size = kernel_size
        self.sigma = sigma

        # Register torch kernels if Kornia is unavailable
        if not USE_KORNIA:
            # Build Gaussian kernel
            img_coords = torch.arange(
                kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
            
            yy, xx = torch.meshgrid(img_coords, img_coords, indexing='ij')
            gaussian_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            self.register_buffer('gauss_kernel', gaussian_kernel.view(
                1, 1, kernel_size, kernel_size))
            
            # Build Laplacian kernel
            laplacian_kernel = torch.tensor(
                [[0., 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            self.register_buffer('lap_kernel', laplacian_kernel.view(1, 1, 3, 3))
        else:
            
            self.kornia_laplacian = KF.Laplacian(kernel_size, border_type='reflect', normalized=True)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Unsqueeze input to (B, C, H, W)
        orig_dim = x.dim()
        x_bchw = _ensure_batch(x)
        x_bchw = ConvertToLuminance(x_bchw)

        if USE_KORNIA:
            # Kornia's gaussian_laplace: input (B,C,H,W)
            log = self.kornia_laplacian(x_bchw)
            return _restore_shape(log, orig_dim)

        else:
            # Use custom kernels
            device = x.device
            gauss = self.gauss_kernel.to(device)
            lap = self.lap_kernel.to(device)
            blur = F.conv2d(x_bchw, gauss, padding=self.kernel_size // 2) # type: ignore
            log = F.conv2d(blur, lap, padding=1) # type: ignore

            return _restore_shape(log, orig_dim)


class DistanceTransformMap(nn.Module):
    """
    Euclidean distance transform of binary mask.
    Non-differentiable (OpenCV backend).
    """

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        orig_dim = mask.dim()
        x_bchw = _ensure_batch(mask)
        device = x_bchw.device

        # Convert to luminance 
        x_bchw = ConvertToLuminance(x_bchw)

        # Scale adjustment: if in [0,1] then multiply by 255   
        if x_bchw.max() <= 1:
            x_bchw = x_bchw * 255.0

        # Convert to uint8 and apply distance transform
        img = torch_to_numpy(tensor=x_bchw).astype(np.uint8)
        out_np = np.stack([cv2.distanceTransform(img_i[0], cv2.DIST_L2, 5) for img_i in img])
        
        out = torch.from_numpy(out_np).to(device).unsqueeze(1)

        return _restore_shape(out, orig_dim)


class LocalVarianceMap(nn.Module):
    """
    Local variance via sliding window.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size

        # Register parameters
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        self.register_buffer('mean_kernel', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dim = x.dim()
        x_bchw = _ensure_batch(x)
        device = x_bchw.device

        # Convert to luminance 
        x_bchw = ConvertToLuminance(x_bchw)

        kernel = self.mean_kernel.to(device)

        # Compute mean and second moment
        mean = F.conv2d(x_bchw, kernel, padding=self.kernel_size // 2) # type: ignore
        sqr_mean = F.conv2d(x_bchw * x_bchw, kernel, padding=self.kernel_size // 2) # type: ignore
        
        # Compute local variance as Var(X) = E[X^2] - (E[X])^2 (second central moment)
        var = sqr_mean - mean * mean

        return _restore_shape(var, orig_dim)

# Functional wrappers for both NumPy and Torch inputs
def Compute_threshold_mask(img, thr: float | None = None, 
                           quantile: float = 0.85, 
                           return_type: Literal['mask', 'masked_image'] = 'mask') -> torch.Tensor | np.ndarray:
    """
    Computes a unified threshold mask.

    Args:
        img: Input image, either a NumPy array (H, W) or (H, W, C), or a PyTorch tensor.
        thr: Absolute threshold value. If None, uses quantile.
        quantile: Quantile value for thresholding (0 < quantile < 1).
        return_type: Specifies the return type, either 'mask' for binary mask 
                     or 'masked_image' for the masked image.

    Returns:
        If input is a NumPy array, returns a NumPy array. If input is a PyTorch tensor, 
        returns a PyTorch tensor.
    """
    # Torch input handling
    if isinstance(img, torch.Tensor):
        operator = QuantileThresholdMask(thr, quantile, return_type)

        return operator(img) # type: ignore
    
    # NumPy case
    array = img
    
    # Detect channels-last (H,W,C)
    if array.ndim == 3 and array.shape[2] in (1, 3):
        arr_ch = array.transpose(2, 0, 1)

    elif array.ndim == 2:
        arr_ch = array[np.newaxis, ...]

    else:
        raise ValueError(f"Unsupported numpy shape {array.shape}. Must be 2D or 3D (no batch dim).")
    
    arr_ch = numpy_to_torch(arr_ch)
    out = QuantileThresholdMask(thr, quantile, return_type)(arr_ch)
    out_np = torch_to_numpy(out)

    # Output interface
    if array.ndim == 2:
        out_np = out_np[0] 
    elif array.ndim == 3 and array.shape[2] in (1, 3):
        # Re-transpose
        out_np = out_np.transpose(1, 2, 0)
    
    return out_np

def Apply_sobel_gradient(img):
    """Applies the Sobel gradient to an input image.

    This function computes the Sobel gradient magnitude feature map for an input
    image. It supports both NumPy arrays and PyTorch tensors as input and returns
    the result in the same type as the input.

    Args:
        img: Input image, either a NumPy array (H, W) or (H, W, C), or a PyTorch tensor.

    Returns:
        The Sobel gradient magnitude feature map as a NumPy array or PyTorch tensor,
        depending on the input type.

    Raises:
        ValueError: If the input image has an unsupported shape.
    """
    if isinstance(img, torch.Tensor):
        return SobelGradient()(img)

    array = img
    if array.ndim == 3 and array.shape[2] in (1, 3):
        arr_ch = array.transpose(2, 0, 1)
    elif array.ndim == 2:
        arr_ch = array[np.newaxis, ...]
    else:
        raise ValueError(
            f"Unsupported numpy shape {array.shape}. Must be 2D or 3D.")
    
    arr_ch = numpy_to_torch(arr_ch)
    out = SobelGradient()(arr_ch)
    out_np = torch_to_numpy(out)

    # Output interface
    if array.ndim == 2:
        out_np = out_np[0]
    elif array.ndim == 3 and array.shape[2] in (1, 3):
        # Re-transpose
        out_np = out_np.transpose(1, 2, 0)

    return out_np

def Apply_laplacian_of_gaussian(img, kernel_size: int = 5, sigma: float = 1.0):
    """
    Unified Laplacian of Gaussian (LoG) computation.

    This function computes the Laplacian of Gaussian feature map for an input
    image. It supports both NumPy arrays and PyTorch tensors as input and returns
    the result in the same type as the input.

    Args:
        img: Input image, either a NumPy array (H, W) or (H, W, C), or a PyTorch tensor.
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        The Laplacian of Gaussian feature map as a NumPy array or PyTorch tensor,
        depending on the input type.

    Raises:
        ValueError: If the input image has an unsupported shape.
    """
    if isinstance(img, torch.Tensor):
        return LaplacianOfGaussian(kernel_size, sigma)(img)

    array = img

    if array.ndim == 3 and array.shape[2] in (1, 3):
        arr_ch = array.transpose(2, 0, 1)
    elif array.ndim == 2:
        arr_ch = array[np.newaxis, ...]
    else:
        raise ValueError(
            f"Unsupported numpy shape {array.shape}. Must be 2D or 3D.")
    
    arr_ch = numpy_to_torch(arr_ch)
    out = LaplacianOfGaussian(kernel_size, sigma)(arr_ch)
    out_np = torch_to_numpy(out)

    # Output interface
    if array.ndim == 2:
        out_np = out_np[0]
    elif array.ndim == 3 and array.shape[2] in (1, 3):
        # Re-transpose
        out_np = out_np.transpose(1, 2, 0)

    return out_np

def Compute_distance_transform_map(img):
    """
    Unified distance transform: accepts np.ndarray or torch.Tensor. Returns same type.
    """
    if isinstance(img, torch.Tensor):
        return DistanceTransformMap()(img)

    array = img
    # Treat multi-channel as single-mask by luminance or first channel

    if array.ndim == 3 and array.shape[2] in (1, 3):
        arr_ch = array.transpose(2, 0, 1)
        arr_ch = arr_ch[0]

    elif array.ndim == 2:
        arr_ch = array

    else:
        raise ValueError(
            f"Unsupported numpy shape {array.shape}. Must be 2D or 3D.")
    
    arr_ch = numpy_to_torch(arr_ch)
    out = DistanceTransformMap()(arr_ch)
    out_np = torch_to_numpy(out)

    return out_np

def Compute_local_variance_map(img, kernel_size: int = 5):
    """
    Computes the local variance map for an input image.

    This function calculates the local variance of an image using a sliding window
    approach. It supports both NumPy arrays and PyTorch tensors as input and returns
    the result in the same type as the input.

    Args:
        img: Input image, either a NumPy array (H, W) or (H, W, C), or a PyTorch tensor.
        kernel_size: Size of the sliding window kernel.

    Returns:
        The local variance map as a NumPy array or PyTorch tensor, depending on the input type.

    Raises:
        ValueError: If the input image has an unsupported shape.
    """
    if isinstance(img, torch.Tensor):
        return LocalVarianceMap(kernel_size)(img)

    array = img
    if array.ndim == 3 and array.shape[2] in (1, 3):
        arr_ch = array.transpose(2, 0, 1)
    elif array.ndim == 2:
        arr_ch = array[np.newaxis, ...]
    else:
        raise ValueError(
            f"Unsupported numpy shape {array.shape}. Must be 2D or 3D.")
    
    arr_ch = numpy_to_torch(arr_ch)
    out = LocalVarianceMap(kernel_size)(arr_ch)
    out_np = torch_to_numpy(out)

    # Output interface
    if array.ndim == 2:
        out_np = out_np[0]
    elif array.ndim == 3 and array.shape[2] in (1, 3):
        # Re-transpose
        out_np = out_np.transpose(1, 2, 0)

    return out_np
