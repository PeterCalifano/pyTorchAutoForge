from pickle import NONE
import cli
from kornia.augmentation import AugmentationSequential
import kornia.augmentation as K
import kornia.geometry as KG
import albumentations
import lazy_loader
from regex import E
import torch
from kornia import augmentation as kornia_aug
from torch import nn
from abc import ABC, abstractmethod
import pytest  # For unit tests
from dataclasses import dataclass
# , cupy # cupy for general GPU acceleration use (without torch) to test
import numpy as np
from enum import Enum

from pyTorchAutoForge.utils.conversion_utils import torch_to_numpy, numpy_to_torch
from pyTorchAutoForge.datasets.DataAugmentation import AugsBaseClass

# %% Configuration dataclasses
ndArrayOrTensor = np.ndarray | torch.Tensor


class RandomGaussianNoiseVariableSigma(nn.Module):
    """
    Applies per-sample Gaussian noise with variable sigma.
    sigma_noise: scalar, (min,max) tuple, or per-sample (B,) or (B,2) array/tensor
    gaussian_noise_aug_prob: probability to apply noise per sample
    """

    def __init__(self, sigma_noise: float | tuple[float, float],
                 gaussian_noise_aug_prob: float = 0.5):
        super().__init__()

        self.sigma_gaussian_noise_dn = sigma_noise
        self.gaussian_noise_aug_prob = gaussian_noise_aug_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        device = x.device
        
        # Determine sigma per sample
        sigma = self.sigma_gaussian_noise_dn

        if isinstance(sigma, tuple):
            min_s, max_s = sigma
            sigma_array = (max_s - min_s) * torch.rand(B, device=device) + min_s
        else:
            sigma_array = torch.full((B,), float(sigma), device=device)
    
        # apply probabilistically per sample
        probability_mask = torch.rand(B, device=device) < self.gaussian_noise_aug_prob

        sigma_array = sigma_array * probability_mask.float()
        sigma_array = sigma_array.view(B, 1, 1, 1)
        noise = torch.randn_like(x) * sigma_array

        return x + noise

@dataclass
class AugmentationConfig:
    # Translation parameters (in pixels)
    max_shift: float | tuple[float, float] = (20.0, 20.0)
    shift_aug_prob: float = 0.5
    # Whether image is normalized (0–1) or raw (0–255)
    is_normalized: bool = True
    # Optional scaling factor. If None, inference attempt based on dtype
    normalization_factor : float | None = None 
    # Optional flag to specify if image is already in the torch layout (overrides guess)
    is_torch_layout: bool | None = None  

    # Gaussian noise
    sigma_gaussian_noise_dn: float | tuple[float, float] = 0.05
    gaussian_noise_aug_prob: float = 0.0

    # Gaussian blur
    kernel_size: tuple[int, int] = (5, 5)
    sigma_gaussian_blur: tuple[float, float] = (0.1, 2.0)
    gaussian_blur_aug_prob: float = 0.0

    # Brightness & contrast
    min_max_brightness_factor: tuple[float, float] = (0.8, 1.2)
    min_max_contrast_factor: tuple[float, float] = (0.8, 1.2)
    brightness_aug_prob: float = 0.0
    contrast_aug_prob: float = 0.0

    # Batch‐mode flag
    batch_size: int | None = None  # inferred at runtime if None

# %% Augmentation helper class

class ImageAugmentationsHelper(torch.nn.Module):
    def __init__(self, augs_cfg: AugmentationConfig):
        super().__init__()
        self.augs_cfg = augs_cfg

        # Define kornia augmentation pipeline
        augs_ops = nn.ModuleList()

        if augs_cfg.brightness_aug_prob > 0:
            # Random brightness scaling
            augs_ops.append(module=K.RandomBrightness(brightness=augs_cfg.min_max_brightness_factor,
                                                      p=augs_cfg.brightness_aug_prob,
                                                      keepdim=True,
                                                      clip_output=False))

        if augs_cfg.contrast_aug_prob > 0:
            # Random contrast scaling
            augs_ops.append(module=K.RandomContrast(contrast=augs_cfg.min_max_contrast_factor,
                                                    p=augs_cfg.contrast_aug_prob,
                                                    keepdim=True,
                                                    clip_output=False))

        if augs_cfg.gaussian_blur_aug_prob > 0:
            # Random Gaussian blur
            augs_ops.append(module=K.RandomGaussianBlur(kernel_size=augs_cfg.kernel_size,
                                                        sigma=augs_cfg.sigma_gaussian_blur,
                                                        p=augs_cfg.gaussian_blur_aug_prob,
                                                        keepdim=True))

        if augs_cfg.gaussian_noise_aug_prob > 0:
            # Random Gaussian noise
            augs_ops.append(module=RandomGaussianNoiseVariableSigma(sigma_noise=augs_cfg.sigma_gaussian_noise_dn, gaussian_noise_aug_prob=augs_cfg.gaussian_noise_aug_prob))
            
        # Stack into nn.Sequential module
        self.kornia_augs_module = nn.Sequential(*augs_ops)

    def forward(self, images: ndArrayOrTensor,
                labels: ndArrayOrTensor
                ) -> tuple[ndArrayOrTensor, ndArrayOrTensor]:
        """
            images: Tensor[B,H,W,C] or [B,C,H,W], or np.ndarray [...,H,W,C]
            labels: Tensor[B, num_points, 2] or np.ndarray matching batch
            returns: shifted+augmented images & labels, same type as input
        """

        # Detect type and convert to torch Tensor [B,C,H,W]
        is_numpy = isinstance(images, np.ndarray)
        img_tensor, to_numpy, scale_factor = self.preprocess_images_(images)

        # Apply translation
        if self.augs_cfg.shift_aug_prob > 0:
            img_shifted, lbl_shifted = self.translate_batch_(img_tensor, labels)
        else:
            img_shifted = img_tensor
            lbl_shifted = numpy_to_torch(labels).float()

        # Apply kornia augmentations
        if scale_factor is not None: # Perhaps just avoid scaling in the preprocess?
            img_shifted = scale_factor * img_shifted

        aug_img = self.kornia_augs_module(img_shifted)

        # Apply inverse scaling if needed
        if scale_factor is not None:
            aug_img = aug_img / scale_factor

        # Apply clamping to [0,1]
        aug_img = torch.clamp(aug_img, 0.0, 1.0)

        # Convert back to numpy if was ndarray
        if to_numpy is True:
            aug_img = torch_to_numpy(aug_img.permute(0, 2, 3, 1))
            lbl_shifted = torch_to_numpy(lbl_shifted)

        return aug_img, lbl_shifted

    def preprocess_images_(self,
                    images: ndArrayOrTensor
                    ) -> tuple[torch.Tensor, bool, float]:
        """
        Preprocess images for augmentation.

        Converts input images (numpy arrays or PyTorch tensors) to a standardized
        PyTorch tensor format [B, C, H, W], applying necessary transformations
        such as layout adjustments, normalization, and dtype conversion.

        Args:
            images (ndArrayOrTensor): Input images as numpy arrays or PyTorch tensors.

        Raises:
            ValueError: If the input image shape is unsupported.
            TypeError: If the input type is neither numpy array nor PyTorch tensor.

        Returns:
            tuple[torch.Tensor, bool]: A tuple containing the processed image tensor
            and a boolean indicating whether the input was originally a numpy array.
        """

        if isinstance(images, np.ndarray):
            
            imgs_array = images.copy()
            dtype = imgs_array.dtype

            if imgs_array.ndim < 2 or imgs_array.ndim > 4:
                raise ValueError("Unsupported image shape. Expected 2D, 3D or 4D array.")

            # If numpy and not specified, assume (B, H, W, C) layout, else use flag
            is_numpy_layout: bool = True if imgs_array.shape[-1] in (1, 3) else False
            if self.augs_cfg.is_torch_layout is not None:
                is_numpy_layout = not self.augs_cfg.is_torch_layout

            # Perform convert, unsqueeze and permutation according to flags 
            imgs_array = torch.from_numpy(imgs_array.astype(np.float32))

            if imgs_array.ndim == 2 and is_numpy_layout:
                # Single grayscale HxW
                imgs_array = imgs_array.unsqueeze(0)  # Expand to (1,H,W)
                imgs_array = imgs_array.unsqueeze(-1)  # Expand to (1,H,W,1)

            elif imgs_array.ndim == 3:
            
                if is_numpy_layout and self.augs_cfg.is_torch_layout is None:
                    # Single color or grayscale image (H,W,C)
                    imgs_array = imgs_array.unsqueeze(0)  # Expand to (1,H,W,C)
                    imgs_array = imgs_array.permute(0, 3, 1, 2) # Permute to (B,C,H,W)

                elif not is_numpy_layout:
                    # Torch layout or multiple batch images
                    if self.augs_cfg.is_torch_layout is None: # Then multiple images, determined by C
                        imgs_array = imgs_array.unsqueeze(1)  # Expand to (B,1,H,W)
                    else:
                        # Multiple grayscale images (B,H,W)
                        imgs_array = imgs_array.unsqueeze(-1)  # Expand to (B,H,W,1)
                        imgs_array = imgs_array.permute(0, 3, 1, 2) # Permute to (B,C,H,W)

            elif imgs_array.ndim == 4:
                if is_numpy_layout:   
                    imgs_array = imgs_array.permute(0, 3, 1, 2)
                # else: If not numpy layout, there is nothing to do

            # Apply normalization if needed
            scale_factor = 1.0 
            if not self.augs_cfg.is_normalized:
                if self.augs_cfg.normalization_factor is not None:
                    scale_factor = self.augs_cfg.normalization_factor  
                else:
                    # Guess based on dtype
                    if dtype == torch.uint8 or dtype == np.uint8:
                        scale_factor = 255.0
                    elif dtype == torch.uint16 or dtype == np.uint16:
                        scale_factor = 65535.0
                    elif dtype == torch.uint32 or dtype == np.uint32:
                        scale_factor = 4294967295.0
                
                imgs_array = imgs_array / scale_factor

            return imgs_array, True, scale_factor
        
        elif torch.is_tensor(images):

            imgs_array = images
            dtype = imgs_array.dtype
            scale_factor = 1.0

            if not self.augs_cfg.is_normalized:
                if self.augs_cfg.normalization_factor is not None:
                    scale_factor = self.augs_cfg.normalization_factor  
                else:
                    # Guess based on dtype
                    if dtype == torch.uint8 or dtype == np.uint8:
                        scale_factor = 255.0
                    elif dtype == torch.uint16 or dtype == np.uint16:
                        scale_factor = 65535.0
                    elif dtype == torch.uint32 or dtype == np.uint32:
                        scale_factor = 4294967295.0
                
                imgs_array = imgs_array / scale_factor
        
            if imgs_array.dim() == 3:
                imgs_array = imgs_array.unsqueeze(0) 

            # Detect [B,H,W,C] vs [B,C,H,W]
            if imgs_array.shape[-1] in (1,3) and imgs_array.dim()==4:
                imgs_array = imgs_array.permute(0,3,1,2) # Detected numpy layout, permute

            return imgs_array, False, scale_factor
        
        else:
            raise TypeError("Unsupported image array type.")

    def translate_batch_(self,
                         images: torch.Tensor,
                         labels: ndArrayOrTensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """
            images: [B,C,H,W]
            labels: torch.Tensor[B,N] or np.ndarray
            returns: shifted images & labels in torch.Tensor
        """

        B, C, H, W = images.shape

        if len(labels.shape) > 2:
            raise NotImplementedError("Current implementation is tailored to translate single point label [Bx2], but got: ", labels.shape)
        
        # Convert labels to tensor [B,N,2]
        lbl = numpy_to_torch(labels).float()
        assert(lbl.shape[0] == B), f"Label batch size {lbl.shape[0]} does not match image batch size {B}."

        # Sample shifts for each batch: dx ∈ [-max_x, max_x], same for dy
        if isinstance(self.augs_cfg.max_shift, (tuple, list)):
            max_x, max_y = self.augs_cfg.max_shift  
        else: 
            max_x, max_y = (self.augs_cfg.max_shift, self.augs_cfg.max_shift)
        
        # Sample shifts by applying 0.99 margin
        dx = 0.99 * torch.randint(-int(max_x), int(max_x)+1, (B,))
        dy = 0.99 * torch.randint(-int(max_y), int(max_y)+1, (B,))

        shifted_imgs = images.new_zeros(images.shape)

        # TODO improve this method, currently not capable of preventing the object to exit the plane
        for i in range(B):
            ox, oy = int(dx[i]), int(dy[i])

            # Apply saturation to avoid out of bounds
            src_x1 = max(0, -ox)
            src_x2 = min(W, W-ox)
            src_y1 = max(0, -oy)
            src_y2 = min(H, H-oy)

            # Compute destination crop coords
            dst_x1 = max(0, ox)
            dst_x2 = dst_x1 + (src_x2-src_x1)
            dst_y1 = max(0, oy)
            dst_y2 = dst_y1 + (src_y2-src_y1)

            # Copy crop in new image
            shifted_imgs[i, :, dst_y1:dst_y2, dst_x1:dst_x2] = images[i, :, src_y1:src_y2, src_x1:src_x2]

            # Shift points labels 
            lbl[i, 0:2] = lbl[i, 0:2] + \
                torch.tensor([ox, oy], dtype=lbl.dtype, device=lbl.device)

        return shifted_imgs, lbl

# %% Prototypes TODO
class ImageNormalizationCoeff(Enum):
    """Enum for image normalization types."""
    SOURCE = -1.0
    NONE = 1.0
    UINT8 = 255.0
    UINT16 = 65535.0
    UINT32 = 4294967295.0


class ImageNormalization():
    """ImageNormalization class.

    This class normalizes image tensors using the specified normalization type.

    Attributes:
        normalization_type (ImageNormalizationType): The type of normalization to apply to the image.
    """

    def __init__(self, normalization_type: ImageNormalizationCoeff = ImageNormalizationCoeff.NONE):
        self.normalization_type = normalization_type

    def __call__(self, image: torch.Tensor) -> torch.Tensor:

        # Check image datatype, if not float, normalize using dtype
        if image.dtype != torch.float32 and image.dtype != torch.float64:

            # Get datatype and select normalization
            if self.normalization_type == ImageNormalizationCoeff.SOURCE and self.normalization_type is not ImageNormalizationCoeff.NONE:

                if image.dtype == torch.uint8:
                    self.normalization_type = ImageNormalizationCoeff.UINT8

                elif image.dtype == torch.uint16:
                    self.normalization_type = ImageNormalizationCoeff.UINT16

                elif image.dtype == torch.uint32:
                    self.normalization_type = ImageNormalizationCoeff.UINT32
                else:
                    raise ValueError(
                        "Normalization type selected as SOURCE but image type is not uint8, uint16 or uint32. Cannot determine normalization value")

        # Normalize image to range [0,1]
        if self.normalization_type.value < 0.0:
            raise ValueError(
                "Normalization for images value cannot be negative.")

        return image / self.normalization_type.value


# TODO GeometryAugsModule
class GeometryAugsModule(AugsBaseClass):
    def __init__(self):
        super(GeometryAugsModule, self).__init__()

        # Example usage
        self.augmentations = AugmentationSequential(
            kornia_aug.RandomRotation(degrees=30.0, p=1.0),
            kornia_aug.RandomAffine(degrees=0, translate=(0.1, 0.1), p=1.0),
            data_keys=["input", "mask"]
        )  # Define the keys: image is "input", mask is "mask"

    def forward(self, x: torch.Tensor, labels: torch.Tensor | tuple[torch.Tensor]) -> torch.Tensor:
        # TODO define interface (input, output format and return type)
        x, labels = self.augmentations(x, labels)

        return x, labels


# %% Deprecated functions
def build_kornia_augs(sigma_noise: float, sigma_gaussian_blur: tuple | float = (0.0001, 1.0),
                      brightness_factor: tuple | float = (0.0001, 0.01),
                      contrast_factor: tuple | float = (0.0001, 0.01)) -> torch.nn.Sequential:

    # Define kornia augmentation pipeline

    # Random brightness
    brightness_min, brightness_max = brightness_factor if isinstance(
        brightness_factor, tuple) else (brightness_factor, brightness_factor)

    random_brightness = kornia_aug.RandomBrightness(brightness=(
        brightness_min, brightness_max), clip_output=False, same_on_batch=False, p=1.0, keepdim=True)

    # Random contrast
    contrast_min, contrast_max = contrast_factor if isinstance(
        contrast_factor, tuple) else (contrast_factor, contrast_factor)

    random_contrast = kornia_aug.RandomContrast(contrast=(
        contrast_min, contrast_max), clip_output=False, same_on_batch=False, p=1.0, keepdim=True)

    # Gaussian Blur
    sigma_gaussian_blur_min, sigma_gaussian_blur_max = sigma_gaussian_blur if isinstance(
        sigma_gaussian_blur, tuple) else (sigma_gaussian_blur, sigma_gaussian_blur)
    gaussian_blur = kornia_aug.RandomGaussianBlur(
        (5, 5), (sigma_gaussian_blur_min, sigma_gaussian_blur_max), p=0.75, keepdim=True)

    # Gaussian noise
    gaussian_noise = kornia_aug.RandomGaussianNoise(
        mean=0.0, std=sigma_noise, p=0.75, keepdim=True)

    # Motion blur
    # direction_min, direction_max = -1.0, 1.0
    # motion_blur = kornia_aug.RandomMotionBlur((3, 3), (0, 360), direction=(direction_min, direction_max), p=0.75, keepdim=True)

    return torch.nn.Sequential(random_brightness, random_contrast, gaussian_blur, gaussian_noise)

def TranslateObjectImgAndPoints(image: torch.Tensor,
                                label: torch.Tensor,
                                max_size_in_pix: float | torch.Tensor | list[float]) -> tuple:

    if not (isinstance(max_size_in_pix, torch.Tensor)):
        max_size_in_pix = torch.Tensor([max_size_in_pix, max_size_in_pix])

    num_entries = 1  # TODO update to support multiple images

    # Get image size
    image_size = image.shape

    # Get max shift coefficients (how many times the size enters half image with margin)
    # TODO validate computation
    max_vertical = 0.99 * (0.5 * image_size[1] / max_size_in_pix[1] - 1)
    max_horizontal = 0.99 * (0.5 * image_size[2] / max_size_in_pix[0] - 1)

    raise NotImplementedError("TODO")

    # Sample shift interval uniformly --> TODO for batch processing: this has to generate uniformly sampled array
    shift_horizontal = torch.randint(-max_horizontal,
                                     max_horizontal, (num_entries,))
    shift_vertical = torch.randint(-max_vertical, max_vertical, (num_entries,))

    # Shift vector --> TODO for batch processing: becomes a matrix
    origin_shift_vector = torch.round(torch.Tensor(
        [shift_horizontal, shift_vertical]) * max_size_in_pix)

    # print("Origin shift vector: ", originShiftVector)

    # Determine index for image cropping
    # Vertical
    idv1 = int(np.floor(np.max([0, origin_shift_vector[1]])))
    idv2 = int(
        np.floor(np.min([image_size[1], origin_shift_vector[1] + image_size[1]])))

    # Horizontal
    idu1 = int(np.floor(np.max([0, origin_shift_vector[0]])))
    idu2 = int(
        np.floor(np.min([image_size[2], origin_shift_vector[0] + image_size[2]])))

    croppedImg = image[:, idv1:idv2, idu1:idu2]

    # print("Cropped image shape: ", croppedImg.shape)

    # Create new image and store crop
    shiftedImage = torch.zeros(
        image_size[0], image_size[1], image_size[2], dtype=torch.float32)

    # Determine index for pasting
    # Vertical
    idv1 = int(abs(origin_shift_vector[1])
               ) if origin_shift_vector[1] < 0 else 0
    idv2 = idv1 + croppedImg.shape[1]
    # Horizontal
    idu1 = int(abs(origin_shift_vector[0])
               ) if origin_shift_vector[0] < 0 else 0
    idu2 = idu1 + croppedImg.shape[2]

    shiftedImage[:, idv1:idv2, idu1:idu2] = croppedImg

    # Shift labels (note that coordinate of centroid are modified in the opposite direction as of the origin)
    shiftedLabel = label - \
        torch.Tensor(
            [origin_shift_vector[0], origin_shift_vector[1], 0], dtype=torch.float32)

    return shiftedImage, shiftedLabel


# Quick visual check for pipeline
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cfg = AugmentationConfig(
                    max_shift=(500, 500),
                    shift_aug_prob=0.35,
                    is_normalized=False,
                    sigma_gaussian_noise_dn=15,
                    gaussian_noise_aug_prob=1.0,
                    gaussian_blur_aug_prob=1.0,
                    is_torch_layout=False,
                    min_max_brightness_factor=(0.2, 1.2),
                    min_max_contrast_factor=(0.2, 1.2),
                    brightness_aug_prob=1.0,
                    contrast_aug_prob=1.0,
                )
    helper = ImageAugmentationsHelper(cfg)

    # create a batch of 4 random grayscale images and simple point labels
    batch_size = 3
    imgs = np.zeros((batch_size, 1024, 1024, 1), dtype=np.uint8)
    center = (512, 512)
    radius = 200

    for i in range(batch_size):
        color_gray = np.random.randint(low=0, high=256)  # Random grayscale color between 0 and 255
        # Draw a white disk in the center of each grayscale image
        y, x = np.ogrid[:1024, :1024]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        imgs[i, mask] = color_gray

    lbls = np.tile(np.array([[512, 512]]), (batch_size, 1))
    out_imgs, out_lbls = helper(numpy_to_torch(imgs), lbls)
    out_imgs = torch_to_numpy(out_imgs.permute(0, 2, 3, 1))

    # plot inputs and outputs in a 2×batch_size grid
    fig, axs = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))

    for i in range(batch_size):
        # input
        axs[0, i].imshow(imgs[i], cmap='gray')
        axs[0, i].scatter(lbls[i, 0], lbls[i, 1], c='r', s=20)
        axs[0, i].set_title(f"Input {i}")
        axs[0, i].axis('off')
        # output (rescale if needed)
        disp = out_imgs[i]

        if not cfg.is_normalized:
            disp = (disp * 255).astype(np.uint8)

        axs[1, i].imshow(disp, cmap='gray')
        axs[1, i].scatter(out_lbls[i, 0], out_lbls[i, 1], c='r', s=20)
        axs[1, i].set_title(f"Output {i}")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

"""
    # For later use: batch warping
    def _translate_batch(self,
                         images: torch.Tensor,
                         labels: ArrayOrTensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        B,C,H,W = images.shape
        # convert labels
        lbl = torch.from_numpy(labels).float() if isinstance(labels, np.ndarray) else labels.float()
        if lbl.dim()==2:
            lbl = lbl.unsqueeze(0)
        # prepare per-sample max shifts
        if isinstance(self.cfg.max_shift, torch.Tensor):
            ms = self.cfg.max_shift.to(images.device).long()
        elif isinstance(self.cfg.max_shift, list):
            ms = torch.tensor(self.cfg.max_shift, device=images.device).long()
        else:
            fx,fy = self.cfg.max_shift if isinstance(self.cfg.max_shift,tuple) else (self.cfg.max_shift,self.cfg.max_shift)
            ms = torch.tensor([[int(fx),int(fy)]]*B, device=images.device)
        # random shifts uniform integer in [-max, max]
        dx = torch.randint(-ms[:,0], ms[:,0]+1, (B,), device=images.device)
        dy = torch.randint(-ms[:,1], ms[:,1]+1, (B,), device=images.device)
        # normalized translation for affine: tx = dx/(W/2), ty = dy/(H/2)
        tx = dx.float()/(W/2)
        ty = dy.float()/(H/2)
        # build theta for each sample: [[1,0,tx],[0,1,ty]]
        theta = torch.zeros((B,2,3), device=images.device, dtype=images.dtype)
        theta[:,0,0] = 1; theta[:,1,1] = 1
        theta[:,0,2] = tx; theta[:,1,2] = ty
        # grid and sample
        grid = F.affine_grid(theta, images.size(), align_corners=False)
        shifted = F.grid_sample(images, grid, padding_mode='zeros', align_corners=False)
        # shift labels in pixel space
        lbl_t = lbl - torch.stack([dx,dy], dim=1).unsqueeze(1).float()
"""