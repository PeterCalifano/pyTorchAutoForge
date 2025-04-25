import matplotlib.pyplot as plt
from pyTorchAutoForge.datasets import ImagesLabelsCachedDataset, AugmentationConfig, ImageAugmentationsHelper
from pyTorchAutoForge.utils.conversion_utils import torch_to_numpy, numpy_to_torch
import numpy as np


def test_synthetic_mask_augmentation():

    cfg = AugmentationConfig(
        max_shift=(250, 250),
        shift_aug_prob=0.35,
        is_normalized=False,
        rotation_angle=(-180, 180),
        rotation_aug_prob=1.0,
        rotation_interp_mode="bilinear",
        sigma_gaussian_noise_dn=15,
        gaussian_noise_aug_prob=1.0,
        gaussian_blur_aug_prob=1.0,
        is_torch_layout=False,
        min_max_brightness_factor=(0.6, 1.2),
        min_max_contrast_factor=(0.6, 1.2),
        brightness_aug_prob=1.0,
        contrast_aug_prob=1.0,
        input_normalization_factor=255.0,
        enable_auto_input_normalization=True,
    )

    augs_helper = ImageAugmentationsHelper(cfg)

    # Create a batch of 4 random grayscale images and simple point labels
    batch_size = 6
    imgs = np.zeros((batch_size, 1024, 1024, 1), dtype=np.uint8)
    center = (512, 512)
    radius = 200

    for i in range(batch_size):
        # Random grayscale color between 0 and 255
        color_gray = np.random.randint(low=0, high=256)
        # Draw a white disk in the center of each grayscale image
        y, x = np.ogrid[:1024, :1024]
        # Asymmetric ellipse radii
        radius_x = radius
        radius_y = radius // 2
        # Create an axis‐aligned elliptical mask
        mask = ((x - center[0])**2 / radius_x**2 + (y - center[1])**2 / radius_y**2) <= 1
        imgs[i, mask] = color_gray

    lbls = np.tile(np.array([[512, 512]]), (batch_size, 1))
    out_imgs, out_lbls = augs_helper(numpy_to_torch(imgs), lbls)
    out_imgs = torch_to_numpy(out_imgs.permute(0, 2, 3, 1))

    # Plot inputs and outputs in a 2×batch_size grid
    fig, axs = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))

    for i in range(batch_size):
        # Input
        axs[0, i].imshow(imgs[i], cmap='gray')
        axs[0, i].scatter(lbls[i, 0], lbls[i, 1], c='r', s=20)
        axs[0, i].set_title(f"Input {i}")
        axs[0, i].axis('off')
        # Output (rescale if needed)
        disp = out_imgs[i]

        if cfg.is_normalized or cfg.enable_auto_input_normalization:
            disp = (disp * 255).astype(np.uint8)
        else:
            disp = disp.astype(np.uint8)

        axs[1, i].imshow(disp, cmap='gray')
        axs[1, i].scatter(out_lbls[i, 0], out_lbls[i, 1], c='r', s=20)
        axs[1, i].set_title(f"Output {i}")
        axs[1, i].axis('off')

    plt.tight_layout()
    #plt.pause(2)
    plt.show()
    plt.close()

def test_sample_images_augmentation(): 
    pass

# %% Manual test calls
if __name__ == '__main__':
    test_synthetic_mask_augmentation()
    #test_sample_images_augmentation()labels=images=