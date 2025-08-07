import torch
from pyTorchAutoForge.datasets.ImagesAugmentation import (
    PoissonShotNoise,
    RandomGaussianNoiseVariableSigma,
    Flip_coords_X,
    Flip_coords_Y,
)
import pytest
import matplotlib.pyplot as plt
from pyTorchAutoForge.datasets import ImagesLabelsCachedDataset, AugmentationConfig, ImageAugmentationsHelper
from pyTorchAutoForge.utils.conversion_utils import torch_to_numpy, numpy_to_torch
import numpy as np
import PIL
from kornia.constants import DataKey


# Auxiliary functions
def load_sample_images(max_num : int = 5):# -> list[Any]:
    # Load images in samples_imgs folder
    import os
    test_data_path = os.path.dirname(__file__)
    sample_imgs_path = os.path.join(test_data_path, "..", ".test_samples/test_images")
    sample_imgs = os.listdir(sample_imgs_path)

    images = []
    for idx, img_file in enumerate(sample_imgs):
        img_path = os.path.join(sample_imgs_path, img_file)
        images.append(np.array(PIL.Image.open(img_path).convert("L")))

        if idx >= max_num:
            break

    assert len(images) > 0, "No images found in the sample_imgs folder."
    assert all(isinstance(img, np.ndarray) for img in images), "All images should be NumPy arrays."

    return images

# %% Augmentation modules-specific tests
def test_random_gaussian_noise_variable_sigma_no_noise():
    # gaussian_noise_aug_prob=0 ⇒ output equals input
    x = torch.randn(5, 3, 4, 4)
    out = RandomGaussianNoiseVariableSigma(
        sigma_noise=2.0, gaussian_noise_aug_prob=0.0
    )(x)
    assert torch.allclose(out, x)

def test_random_gaussian_noise_variable_sigma_zero_sigma():
    # sigma_noise=0 & prob=1 ⇒ output equals input
    x = torch.randn(2, 1, 3, 3)
    out = RandomGaussianNoiseVariableSigma(
        sigma_noise=0.0, gaussian_noise_aug_prob=1.0
    )(x)
    assert torch.allclose(out, x)

def test_random_gaussian_noise_variable_sigma_shape_dtype():
    # shape and dtype preserved
    x = torch.randn(7, 2, 5, 5, dtype=torch.float32)
    out = RandomGaussianNoiseVariableSigma(
        sigma_noise=(0.1, 0.5), gaussian_noise_aug_prob=1.0
    )(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


@pytest.mark.parametrize("coords,width", [
    (torch.tensor([[0, 1], [3, 4]]), 5),
    (torch.tensor([[[0, 0], [2, 2]], [[1, 1], [4, 4]]]), 6),
])
def test_flip_coords_x(coords, width):
    flipped = Flip_coords_X(coords, image_width=width)
    # original x + flipped x = width - 1
    x_orig = coords[..., 0]
    x_flip = flipped[..., 0]
    assert torch.all(x_orig + x_flip == width - 1)
    # y unchanged
    assert torch.equal(flipped[..., 1], coords[..., 1])


@pytest.mark.parametrize("coords,height", [
    (torch.tensor([[0, 1], [3, 4]]), 5),
    (torch.tensor([[[0, 0], [2, 2]], [[1, 1], [4, 4]]]), 6),
])
def test_flip_coords_y(coords, height):
    flipped = Flip_coords_Y(coords, image_height=height)
    # original y + flipped y = height - 1
    y_orig = coords[..., 1]
    y_flip = flipped[..., 1]
    assert torch.all(y_orig + y_flip == height - 1)
    # x unchanged
    assert torch.equal(flipped[..., 0], coords[..., 0])


def test_flip_coords_invalid_dim():
    bad = torch.randn(2, 2,2,2)
    with pytest.raises(ValueError):
        Flip_coords_X(bad, image_width=10)
    with pytest.raises(ValueError):
        Flip_coords_Y(bad, image_height=10)
    
# %% Integrated tests
def test_synthetic_mask_augmentation():

    augs_datakey = [DataKey.IMAGE, DataKey.KEYPOINTS]
    lbl_datakey = DataKey.KEYPOINTS

    cfg = AugmentationConfig(
        max_shift_img_fraction=(0.5,0.5),
        input_data_keys = augs_datakey,
        shift_aug_prob=0.35,
        is_normalized=False,
        rotation_angle=(-180, 180),
        rotation_aug_prob=1.0,
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
    out_imgs, out_lbls = augs_helper(numpy_to_torch(imgs), numpy_to_torch(lbls))
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
    # Load sample images
    imgs = load_sample_images()    

    # Make all images of same type and size
    resolution = (1024, 1024)
    imgs = [np.array(PIL.Image.fromarray(img).resize(resolution, resample=PIL.Image.LANCZOS)) for img in imgs]

    # Downscale to uint8 if larger
    for i in range(len(imgs)):
        if imgs[i].dtype != np.uint8 and imgs[i].max() >= 255:
            imgs[i] = (imgs[i] / 255.0).astype(np.uint8)

        elif imgs[i].dtype != np.uint8:
            imgs[i] = imgs[i].astype(np.uint8)

    # Convert to batch numpy array
    batch_size = len(imgs)
    imgs = np.stack(imgs, axis=0)

    # Define augmentation helper
    augs_datakey = [DataKey.IMAGE, DataKey.KEYPOINTS]
    lbl_datakey = DataKey.KEYPOINTS

    cfg = AugmentationConfig(
        max_shift_img_fraction=(0.5,0.5),
        input_data_keys = augs_datakey,
        shift_aug_prob=0.35,
        is_normalized=False,
        rotation_angle=(-180, 180),
        rotation_aug_prob=1.0,
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

    # Apply augs and plot
    lbls = np.tile(np.array([resolution[0] / 2, resolution[1] / 2]), (batch_size, 1))

    num_trials = 10

    for idTrial in range(num_trials):

        out_imgs, out_lbls = augs_helper( numpy_to_torch(imgs, dtype=torch.float32), numpy_to_torch(lbls, dtype=torch.float32) )
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
        # plt.pause(2)
        plt.show()
        plt.close()

def test_AugmentationSequential():
    from kornia.augmentation import AugmentationSequential, RandomAffine, RandomHorizontalFlip
    from kornia.constants import DataKey

    # Get sample image
    imgs = load_sample_images()   

    data_augmentation_module = AugmentationSequential(
        RandomAffine(degrees=0.0, 
                     translate=(0.4, 0.4), 
                     p=1.0,
                     keepdim=True),
        RandomHorizontalFlip(p=1.0),
        data_keys=[DataKey.IMAGE, DataKey.MASK, DataKey.KEYPOINTS],
        same_on_batch=False
    )

    def augment_data_batch(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:

        # Apply augmentations
        outputs = data_augmentation_module(*inputs)
        # NOTE: output is a list not a tuple

        # Ensure outputs are in tuple format
        #if not isinstance(outputs, tuple):
        #    outputs = (outputs,)

        return outputs
    
    img = imgs[1]
    image_tensor = numpy_to_torch(img, dtype=torch.float32)
    bin_mask = image_tensor > 0.5  # Example binary mask
    keypoints = torch.tensor(
        # Example keypoints
        [[512, 512], [750, 750]], dtype=torch.float32) # Note that kornia uses the points in the (N,2) format (assuming 1 image!)

    augmented_image, bin_mask, aug_keypoints = augment_data_batch(
        image_tensor, bin_mask, keypoints)

    # Show image and transformed image
    import matplotlib.pyplot as plt

    # Convert tensors to numpy arrays for display
    orig_img = img
    aug_img = augmented_image.squeeze().detach().cpu().numpy()

    if aug_img.ndim == 3 and aug_img.shape[0] in [1, 3]:
        # Move channel to last if needed
        aug_img = np.transpose(aug_img, (1, 2, 0))

    if orig_img.ndim == 3 and orig_img.shape[2] == 1:
        orig_img = orig_img.squeeze(-1)

    if aug_img.ndim == 3 and aug_img.shape[2] == 1:
        aug_img = aug_img.squeeze(-1)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # Plot original image
    axs[0].imshow(orig_img, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Plot original point
    axs[0].scatter(keypoints[0, 0].item(), keypoints[0, 0].item(), c='r', s=40, label='Original Point')
    axs[0].scatter(keypoints[1, 1].item(), keypoints[1, 1].item(), c='g', s=40, label='Original Point 2')

    # Plot augmented image
    axs[1].imshow(aug_img, cmap='gray')
    axs[1].set_title("Augmented Image")
    axs[1].axis('off')

    # Plot transformed point if available
    aug_point = aug_keypoints.detach().cpu().numpy()
    axs[1].scatter(keypoints[0, 0].item(), keypoints[0, 1].item(), c='r', s=40, label='Original Point 1')
    axs[1].scatter(keypoints[1, 0].item(), keypoints[1, 1].item(), c='g', s=40, label='Original Point 2')

    axs[1].scatter(aug_point[0, 0], aug_point[0, 1], c='r', s=40, 
                   label='Transformed Point 1', marker='x')
    axs[1].scatter(aug_point[1, 0], aug_point[1, 1], c='g',
                   s=40, label='Transformed Point 2', marker='x')

    plt.tight_layout()
    plt.show()


# Chain parametrize to test combinations
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shift_aug_prob", [0, 1])
@pytest.mark.parametrize("rotation_aug_prob", [0, 1])
@pytest.mark.parametrize("gaussian_noise_aug_prob", [0, 1])
@pytest.mark.parametrize("gaussian_blur_aug_prob", [0, 1])
@pytest.mark.parametrize("brightness_aug_prob", [0, 1])
@pytest.mark.parametrize("contrast_aug_prob", [0, 1])
def test_augmentation_helper_preserves_device(device, 
                                              shift_aug_prob,
                                              rotation_aug_prob,
                                              gaussian_noise_aug_prob,
                                              gaussian_blur_aug_prob,
                                              brightness_aug_prob,
                                              contrast_aug_prob) -> None:
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    # Create a minimal configuration for the augmentations helper.
    cfg = AugmentationConfig(
        max_shift_img_fraction=(0.8, 0.8),
        input_data_keys=[DataKey.IMAGE, DataKey.KEYPOINTS],
        shift_aug_prob=shift_aug_prob,
        is_normalized=False,
        rotation_angle=(-10, 10),
        rotation_aug_prob=rotation_aug_prob,
        sigma_gaussian_noise_dn=0.0,       # Disable noise for simplicity
        gaussian_noise_aug_prob=gaussian_noise_aug_prob,
        gaussian_blur_aug_prob=gaussian_blur_aug_prob,
        is_torch_layout=False,
        min_max_brightness_factor=(1.0, 1.0),
        min_max_contrast_factor=(1.0, 1.0),
        brightness_aug_prob=brightness_aug_prob,
        contrast_aug_prob=contrast_aug_prob,
        input_normalization_factor=255.0,
        enable_auto_input_normalization=False,
    )

    augs_helper = ImageAugmentationsHelper(cfg)

    # Create dummy inputs on CUDA
    x = torch.randn(1, 1, 64, 64, device=device)
    lbl = torch.tensor([[32, 32]], dtype=torch.float32, device=device)

    out_img, out_lbl = augs_helper(x, lbl)

    # Verify that outputs remain on cuda
    assert out_img.device.type == device
    assert out_lbl.device.type == device

def test_zero_augs_input_unchanged():
    device = 'cpu'
    cfg = AugmentationConfig(
        max_shift_img_fraction=(0.8, 0.8),
        input_data_keys=[DataKey.IMAGE, DataKey.KEYPOINTS],
        shift_aug_prob=0.0,
        is_normalized=True,
        rotation_angle=(-10, 10),
        rotation_aug_prob=0.0,
        sigma_gaussian_noise_dn=0.0,       # Disable noise for simplicity
        gaussian_noise_aug_prob=0.0,
        gaussian_blur_aug_prob=0.0,
        is_torch_layout=True,
        min_max_brightness_factor=(1.0, 1.0),
        min_max_contrast_factor=(1.0, 1.0),
        brightness_aug_prob=0.0,
        contrast_aug_prob=0.0,
        input_normalization_factor=1.0,
        enable_auto_input_normalization=False,
    )

    augs_helper = ImageAugmentationsHelper(cfg)

    # Create dummy inputs
    x = torch.clamp(torch.randn(1, 1, 64, 64, device=device), 0, 1)
    lbl = torch.tensor([[32, 32]], dtype=torch.float32, device=device)

    out_img, out_lbl = augs_helper(x, lbl)

    # Verify that outputs are the same as inputs
    assert torch.allclose(out_img, x), "Output image should be the same as input"
    assert torch.allclose(out_lbl, lbl), "Output labels should be the same as input labels"

# %% MANUAL TEST CALLS
if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    shift_aug_prob = 0.0
    rotation_aug_prob = 0.0
    gaussian_noise_aug_prob = 0.0
    gaussian_blur_aug_prob = 0.0
    brightness_aug_prob = 0.0
    contrast_aug_prob = 0.0


    #test_synthetic_mask_augmentation()
    test_sample_images_augmentation()
    #test_AugmentationSequential()
    #test_augmentation_helper_preserves_device(device,
    #                                          shift_aug_prob,
    #                                          rotation_aug_prob,
    #                                          gaussian_noise_aug_prob,
    #                                          gaussian_blur_aug_prob,
    #                                          brightness_aug_prob,
    #                                          contrast_aug_prob)
    test_zero_augs_input_unchanged()
