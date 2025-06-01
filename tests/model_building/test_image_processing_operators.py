import os
from pyTorchAutoForge.datasets import AugmentationConfig, ImageAugmentationsHelper
import numpy as np
import cv2
import torch
from pyTorchAutoForge.model_building.backbones.image_processing_operators import Compute_threshold_mask, Apply_sobel_gradient, Apply_laplacian_of_gaussian, Compute_distance_transform_map, Compute_local_variance_map
from pyTorchAutoForge.utils import torch_to_numpy, numpy_to_torch

from PIL import Image
import matplotlib.pyplot as plt

def _run_image_operators(image: np.ndarray | torch.Tensor):# -> tuple[Tensor | ndarray[Any, Any], Any | NDArray[numpy_typ...:
    """
    Test image processing operators on input image.
    """

    # Run operators
    mask = Compute_threshold_mask(image)
    grad = Apply_sobel_gradient(image)
    log = Apply_laplacian_of_gaussian(image)
    dist_map = Compute_distance_transform_map(mask)
    var_map = Compute_local_variance_map(image)

    return mask, grad, log, dist_map, var_map


def run_all_single_image_(image_names, apply_augs, augmentation_module: ImageAugmentationsHelper | None = None):

    this_file_path = os.path.dirname(os.path.abspath(__file__))

    for image_name in image_names:
        # ---- Load images ----
        fn = os.path.join(this_file_path, '..', '.test_samples',
                          'test_images', f'{image_name}.png')
        image = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot load image {fn}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize images
        scale = 255.0 if image.dtype == np.uint8 else 2**16
        image_norm = image.astype(np.float32) / scale

        # Optional augmentations
        Y = torch.Tensor([image.shape[0], image.shape[1]]).reshape(1, 2) / 2
        
        if apply_augs and augmentation_module is not None:
            # Convert to torch tensor
            image_norm_torch = numpy_to_torch(image_norm)
            image_norm_torch = image_norm_torch.unsqueeze(0).unsqueeze(0)
            img_aug, Y = augmentation_module(image_norm_torch, Y)
        else:
            img_aug = numpy_to_torch(image_norm)
            img_aug = img_aug.unsqueeze(0).unsqueeze(0)
        Y = torch_to_numpy(Y)

        print(
            f"Testing image {image_name}: min {image_norm.min()}, max {image_norm.max()}, mean {image_norm.mean()}")

        # run operators
        numpy_maps = _run_image_operators(img_aug[0, 0].cpu().numpy())
        torch_maps = _run_image_operators(numpy_to_torch(img_aug).to('cuda'))

        # Compare numpy vs torch maps for unit test
        for i, (nm, tm) in enumerate(zip(numpy_maps, torch_maps)):
            tm_np = tm[0, 0].cpu().numpy()
            assert np.allclose(
                nm, tm_np, atol=1e-5), f"Mismatch in map {i+1} for image {image_name}"
            print(f"Image {image_name} - map {i+1} matches.")

        # Plot results in a 2×3 grid
        titles = ["Original", "Threshold",
                  "Sobel", "LoG", "Distance", "Variance"]

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        axes[0].imshow(image_norm, cmap='gray')
        axes[0].set_title(titles[0])
        axes[0].axis('off')

        for i, tm in enumerate(torch_maps, 1):

            axes[i].imshow(tm[0, 0].cpu().numpy(), cmap='gray')
            axes[i].set_title(titles[i])
            axes[i].axis('off')

        for ax in axes[len(torch_maps)+1:]:
            ax.axis('off')

        fig.suptitle(f"Feature maps for {image_name}")
        plt.tight_layout()

    plt.show()
    plt.close()


def _run_all_batched_images_(image_names, apply_augs, augmentation_module: ImageAugmentationsHelper | None = None):
    
    this_file_path = os.path.dirname(os.path.abspath(__file__))

    # ---- Load and normalize all images ----
    imgs = []
    for name in image_names:
        fn = os.path.join(this_file_path, '..', '.test_samples',
                          'test_images', f'{name}.png')
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image {fn}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img.astype(np.float32))

    # ensure same H×W
    shapes = {im.shape for im in imgs}
    if len(shapes) != 1:
        raise ValueError(
            "All images must have the same dimensions for batching")
    H, W = shapes.pop()

    batch_np = np.stack(imgs, axis=0)  # (N, H, W)
    #scale = 255.0 if batch_np.dtype == np.uint8 else 2**16
    #batch_np /= scale

    # print per-image stats
    for name, im in zip(image_names, batch_np):
        print(
            f"Testing image {name}: min {im.min():.4f}, max {im.max():.4f}, mean {im.mean():.4f}")

    # ---- Build torch batch ----
    batch_torch = numpy_to_torch(batch_np).unsqueeze(1).to('cuda')  # (N,1,H,W)

    # Optional augmentations (all at once)
    if apply_augs and augmentation_module is not None:
        # Y: one (H/2, W/2) center per image
        Y = torch.Tensor([[H, W]] * len(image_names)) / 2  # (N, 2)
        batch_torch, Y = augmentation_module(batch_torch/255.0, Y)
        Y = torch_to_numpy(Y)

    # Run non-batched for numpy (no support for batch)
    numpy_inputs = batch_np if not apply_augs else batch_torch[:, 0].cpu(
    ).numpy()
    numpy_maps_per_image = [_run_image_operators(im) for im in numpy_inputs]
    # Number of maps
    n_maps = len(numpy_maps_per_image[0])
    # stack per-map across images: list length n_maps of arrays (N,H,W)
    numpy_maps = [
        np.stack([maps[i] for maps in numpy_maps_per_image], axis=0)
        for i in range(n_maps)
    ]

    # ---- Run operators in batch ----
    torch_maps = _run_image_operators(batch_torch)

    # ---- Verify all images/maps at once ----
    for i, (nm, tm) in enumerate(zip(numpy_maps, torch_maps), start=1):
        tm_np = tm[:, 0].cpu().numpy()
        assert np.allclose(nm, tm_np, atol=1e-6), f"Mismatch in map {i}"
        print(f"Map {i} matches for all {len(image_names)} images.")

    # ---- Plot a grid: one row per image, one column per map (+ original) ----
    titles = ["Original", "Threshold", "Sobel", "LoG", "Distance", "Variance"]
    n_maps = len(torch_maps)
    n_imgs = len(image_names)

    fig, axes = plt.subplots(n_imgs, n_maps + 1,
                             figsize=(4*(n_maps+1), 4*n_imgs))
    # if only one image, make axes 2D
    if n_imgs == 1:
        axes = axes[np.newaxis, :]

    for r in range(n_imgs):
        # original
        ax = axes[r, 0]
        ax.imshow(batch_np[r], cmap='gray')
        ax.set_title(titles[0])
        ax.axis('off')
        # feature maps
        for c in range(n_maps):
            ax = axes[r, c+1]
            ax.imshow(255*torch_maps[c][r, 0].cpu().numpy(), cmap='gray')
            if r == 0:
                ax.set_title(titles[c+1])
            ax.axis('off')

    fig.suptitle("Feature maps for batched images", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()

def test_all_operators():
    # ---- Configuration ----
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    image_names = ['000002', '000003']  # extend this list for N images
    apply_augs = True

    # common augmentation setup
    cfg = AugmentationConfig(
        max_shift=(150, 150),
        is_normalized=True,
        shift_aug_prob=0.75,
        rotation_angle=(-179, 179),
        rotation_aug_prob=1.0,
        rotation_expand=False,
        sigma_gaussian_noise_dn=10.0,
        gaussian_noise_aug_prob=1.0,
        gaussian_blur_aug_prob=0.1,
        is_torch_layout=True,
        min_max_brightness_factor=(0.8, 1.2),
        min_max_contrast_factor=(0.8, 1.2),
        brightness_aug_prob=1.0,
        contrast_aug_prob=1.0,
        input_normalization_factor=255.0
    )
    augmentation_module = ImageAugmentationsHelper(cfg)
    
    # Tests
    #run_all_single_image_(image_names, apply_augs, augmentation_module)
    _run_all_batched_images_(image_names, apply_augs, augmentation_module)


# Manual test execution
if __name__ == '__main__':
    test_all_operators()
    


