import os
import torch
from pyTorchAutoForge.datasets.ImagesAugmentation import (
    PoissonShotNoise,
    RandomGaussianNoiseVariableSigma,
    BorderAwareRandomAffine,
    RandomSoftBinarizeImage,
)
import pytest
import matplotlib.pyplot as plt
from pyTorchAutoForge.datasets import ImagesLabelsCachedDataset, AugmentationConfig, ImageAugmentationsHelper
from pyTorchAutoForge.utils.conversion_utils import torch_to_numpy, numpy_to_torch
import numpy as np
from time import perf_counter
import PIL
from kornia.constants import DataKey

# Determine matplotlib backend based on environment
import matplotlib
if os.getenv("DISPLAY", "") == "":
    # Use non-interactive backend if no display is available
    matplotlib.use('agg')

# Helper functions


def _load_sample_images(max_num: int = 5):  # -> list[Any]:
    # Load images in samples_imgs folder
    import os
    test_data_path = os.path.dirname(__file__)
    sample_imgs_path = os.path.join(
        test_data_path, "..", ".test_samples/test_images")

    sample_imgs = os.listdir(sample_imgs_path)

    # Apply random shuffling of sample_imgs
    np.random.shuffle(sample_imgs)

    images = []
    for idx, img_file in enumerate(sample_imgs):
        if idx >= max_num:
            break

        img_path = os.path.join(sample_imgs_path, img_file)
        images.append(np.array(PIL.Image.open(img_path).convert("L")))

    assert len(images) > 0, "No images found in the sample_imgs folder."
    assert all(isinstance(img, np.ndarray)
               for img in images), "All images should be NumPy arrays."

    return images


def _assert_module_device(module, device: torch.device) -> None:
    """Helper function to assert that a nn.Module is on a specific device"""
    for name, param in module.named_parameters(recurse=False):
        assert param.device.type == device.type, (
            f"{module.__class__.__name__}.{name} on {param.device}, expected {device}."
        )

    for name, buf in module.named_buffers(recurse=False):
        assert buf.device.type == device.type, (
            f"{module.__class__.__name__}.{name} on {buf.device}, expected {device}."
        )

    mod_device = getattr(module, "device", None)
    if isinstance(mod_device, torch.device):
        assert mod_device.type == device.type, (
            f"{module.__class__.__name__}.device is {mod_device}, expected {device}."
        )

    param_gen = getattr(module, "_param_generator", None)
    if param_gen is not None:
        gen_device = getattr(param_gen, "device", None)
        if isinstance(gen_device, torch.device):
            assert gen_device.type == device.type, (
                f"{module.__class__.__name__}._param_generator.device is {gen_device}, expected {device}."
            )


def _assert_helper_modules_on_device(augs_helper: ImageAugmentationsHelper,
                                     device: torch.device,
                                     ) -> None:
    """Helper function to assert that modules loaded in ImageAugmentationsHelper are on a specific device"""
    if augs_helper.kornia_augs_module is not None:
        for module in augs_helper.kornia_augs_module.children():
            _assert_module_device(module, device)

    if augs_helper.torchvision_augs_module is not None:
        for module in augs_helper.torchvision_augs_module.children():
            _assert_module_device(module, device)


def _time_augmentation_helper(augs_helper: ImageAugmentationsHelper,
                              images: torch.Tensor,
                              labels: torch.Tensor,
                              num_iters: int = 100,
                              ) -> tuple[float, float]:
    """Helper function to time execution of ImageAugmentationsHelper forward pass (compare cpu vs gpu)"""
    for _ in range(3):
        augs_helper(images, labels)

    if images.device.type == "cuda":
        torch.cuda.synchronize()

    start = perf_counter()
    for _ in range(num_iters):
        augs_helper(images, labels)

    if images.device.type == "cuda":
        torch.cuda.synchronize()

    return perf_counter() - start, (perf_counter() - start) / num_iters

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


def test_random_gaussian_noise_validation_filters_dark_images():
    # Validation should zero sigma for images without enough bright pixels
    torch.manual_seed(0)
    bright_img = torch.full((1, 1, 24, 24), 10.0)
    dark_img = torch.zeros((1, 1, 24, 24))
    half_img = torch.zeros((1, 1, 24, 24))

    # Add some content in half_img
    half_img[0, 0, 4:8, 9:17] = 1.0

    x = torch.cat([bright_img, dark_img, half_img], dim=0)

    aug = RandomGaussianNoiseVariableSigma(sigma_noise=1.0,
                                           gaussian_noise_aug_prob=1.0,
                                           keep_scalar_sigma_fixed=True,
                                           enable_img_validation_mode=True,
                                           validation_min_num_bright_pixels=10,
                                           validation_pixel_threshold=15.0,  # Force auto-adjust branch
                                           )

    out = aug(x)

    # Noise applied only to the bright image; dark image remains unchanged
    assert not torch.allclose(out[0], x[0])
    assert torch.allclose(out[1], x[1])
    assert not torch.allclose(out[2], x[2])


def test_detect_border_crossing_white_masks():
    imgs = torch.zeros((4, 1, 16, 16), dtype=torch.float32)

    # No crossing: bright square away from borders
    imgs[0, 0, 6:10, 6:10] = 255.0

    # Vertical crossing: left border run
    imgs[1, 0, 3:9, 0] = 255.0

    # Horizontal crossing: top border run
    imgs[2, 0, 0, 4:10] = 255.0

    # Both crossings: top + left
    imgs[3, 0, 0, 2:8] = 255.0
    imgs[3, 0, 2:8, 0] = 255.0

    crossing_type, vertical, horizontal = BorderAwareRandomAffine.Detect_border_crossing(
        imgs, num_pix_detect=4, intensity_threshold_uint8=7.0)

    assert crossing_type.tolist() == [0, 1, 2, 3]
    assert vertical.squeeze(-1).tolist() == [False, True, False, True]
    assert horizontal.squeeze(-1).tolist() == [False, False, True, True]

# %% Integrated tests


def test_synthetic_mask_augmentation():

    augs_datakey = [DataKey.IMAGE, DataKey.KEYPOINTS]
    lbl_datakey = DataKey.KEYPOINTS

    cfg = AugmentationConfig(
        max_shift_img_fraction=(0.5, 0.5),
        input_data_keys=augs_datakey,
        shift_aug_prob=0.35,
        is_normalized=False,
        rotation_angle=(-180, 180),
        rotation_aug_prob=1.0,
        sigma_gaussian_noise_dn=15,
        gaussian_noise_aug_prob=1.0,
        gaussian_blur_aug_prob=1.0,
        softbinarize_aug_prob=0.5,
        softbinarize_thr_quantile=0.005,
        softbinarize_blending_factor_minmax=(0.3, 0.7),
        is_torch_layout=False,
        min_max_brightness_factor=(0.6, 1.2),
        min_max_contrast_factor=(0.6, 1.2),
        brightness_aug_prob=1.0,
        contrast_aug_prob=1.0,
        input_normalization_factor=255.0,
        enable_auto_input_normalization=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
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
        mask = ((x - center[0])**2 / radius_x**2 +
                (y - center[1])**2 / radius_y**2) <= 1
        imgs[i, mask] = color_gray

    lbls = np.tile(np.array([[512, 512]]), (batch_size, 1))
    out_imgs, out_lbls = augs_helper(
        numpy_to_torch(imgs), numpy_to_torch(lbls))
    out_imgs = torch_to_numpy(out_imgs.permute(0, 2, 3, 1))
    out_lbls = torch_to_numpy(out_lbls)

    # Plot inputs and outputs in a 2×batch_size grid
    fig, axs = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))
    test_name = "Synthetic mask augs"

    # Set figure name
    fig.suptitle(test_name, fontsize=16)

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


def test_sample_images_augmentation():
    # Load sample images
    imgs = _load_sample_images()

    # Zero out everything that is below 1
    for i in range(len(imgs)):
        imgs[i][imgs[i] < 1] = 0

    # Make all images of same type and size
    resolution = (1024, 1024)
    imgs = [np.array(PIL.Image.fromarray(img).resize(
        resolution, resample=PIL.Image.LANCZOS)) for img in imgs]

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

    cfg = AugmentationConfig(max_shift_img_fraction=(0.5, 0.5),
                             input_data_keys=augs_datakey,
                             shift_aug_prob=0.35,
                             is_normalized=False,
                             rotation_angle=(-180, 180),
                             rotation_aug_prob=1.0,
                             sigma_gaussian_noise_dn=15,
                             gaussian_noise_aug_prob=1.0,
                             gaussian_blur_aug_prob=1.0,
                             softbinarize_aug_prob=0.5,
                             softbinarize_thr_quantile=0.01,
                             softbinarize_blending_factor_minmax=(0.3, 0.7),
                             is_torch_layout=False,
                             min_max_brightness_factor=(0.6, 1.2),
                             min_max_contrast_factor=(0.6, 1.2),
                             brightness_aug_prob=1.0,
                             contrast_aug_prob=1.0,
                             input_normalization_factor=255.0,
                             enable_auto_input_normalization=True,
                             enable_batch_validation_check=True,
                             device='cuda' if torch.cuda.is_available() else 'cpu'
                             )

    augs_helper = ImageAugmentationsHelper(cfg)

    # Apply augs and plot
    lbls = np.tile(
        np.array([resolution[0] / 2, resolution[1] / 2]), (batch_size, 1))

    num_trials = 10

    for idTrial in range(num_trials):

        out_imgs, out_lbls = augs_helper(numpy_to_torch(
            imgs, dtype=torch.float32), numpy_to_torch(lbls, dtype=torch.float32))
        out_imgs = torch_to_numpy(out_imgs.permute(0, 2, 3, 1))
        out_lbls = torch_to_numpy(out_lbls)

        # Plot inputs and outputs in a 2×batch_size grid
        fig, axs = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))
        test_name = "Sample images augmentation (full module)"
        # Set figure name
        fig.suptitle(test_name, fontsize=16)

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
        # Call show only if gui is available
        if plt.get_backend() != 'agg':
            plt.show()
        plt.close()


def test_random_softbinarize_only_sample_images_with_viz():
    # Load sample images
    np.random.seed(0)
    torch.manual_seed(0)
    imgs = _load_sample_images()

    # Make all images of same type and size
    resolution = (1024, 1024)
    imgs = [np.array(PIL.Image.fromarray(img).resize(
        resolution, resample=PIL.Image.LANCZOS)) for img in imgs]

    # Downscale to uint8 if larger
    for i in range(len(imgs)):
        if imgs[i].dtype != np.uint8 and imgs[i].max() >= 255:
            imgs[i] = (imgs[i] / 255.0).astype(np.uint8)

        elif imgs[i].dtype != np.uint8:
            imgs[i] = imgs[i].astype(np.uint8)

    # Convert to batch numpy array
    batch_size = len(imgs)
    imgs = np.stack(imgs, axis=0)

    # Zero out everything that is below 2
    imgs[imgs < 2] = 0

    imgs_t = numpy_to_torch(imgs, dtype=torch.float32)
    if imgs_t.ndim == 3:
        imgs_t = imgs_t.unsqueeze(1)

    aug = RandomSoftBinarizeImage(
        aug_prob=1.0,
        masking_quantile=0.01,
        blending_factor_minmax=(0.1, 0.9),
    )

    out_imgs = aug(imgs_t)

    assert out_imgs.shape == imgs_t.shape
    assert out_imgs.dtype == imgs_t.dtype
    per_sample_delta = (
        out_imgs - imgs_t).abs().view(batch_size, -1).max(dim=1).values
    assert torch.any(per_sample_delta > 0)

    out_imgs = torch_to_numpy(out_imgs)
    if out_imgs.ndim == 4:
        out_imgs = out_imgs[:, 0, :, :]

    fig, axs = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))
    test_name = "RandomSoftBinarizeImage only (sample images)"
    fig.suptitle(test_name, fontsize=16)

    for i in range(batch_size):
        # Input
        axs[0, i].imshow(imgs[i], cmap='gray')
        axs[0, i].set_title(f"Input {i}")
        axs[0, i].axis('off')

        # Output
        disp = out_imgs[i]
        if disp.max() <= 1.0:
            disp = (disp * 255).astype(np.uint8)
        else:
            disp = disp.astype(np.uint8)

        axs[1, i].imshow(disp, cmap='gray')
        axs[1, i].set_title(f"SoftBinarize {i}")
        axs[1, i].axis('off')

    plt.tight_layout()
    if plt.get_backend() != 'agg':
        plt.show()
    plt.close()


def test_AugmentationSequential():
    from kornia.augmentation import AugmentationSequential, RandomAffine, RandomHorizontalFlip
    from kornia.constants import DataKey

    # Get sample image
    imgs = _load_sample_images()

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
        return outputs

    img = imgs[1]
    image_tensor = numpy_to_torch(img, dtype=torch.float32)
    bin_mask = image_tensor > 0.5  # Example binary mask
    keypoints = torch.tensor(
        # Example keypoints
        [[512, 512], [750, 750]], dtype=torch.float32)  # Note that kornia uses the points in the (N,2) format (assuming 1 image!)

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
    axs[0].scatter(keypoints[0, 0].item(), keypoints[0, 0].item(),
                   c='r', s=40, label='Original Point')
    axs[0].scatter(keypoints[1, 1].item(), keypoints[1, 1].item(),
                   c='g', s=40, label='Original Point 2')

    # Plot augmented image
    axs[1].imshow(aug_img, cmap='gray')
    axs[1].set_title("Augmented Image")
    axs[1].axis('off')

    # Plot transformed point if available
    aug_point = aug_keypoints.detach().cpu().numpy()
    axs[1].scatter(keypoints[0, 0].item(), keypoints[0, 1].item(),
                   c='r', s=40, label='Original Point 1')
    axs[1].scatter(keypoints[1, 0].item(), keypoints[1, 1].item(),
                   c='g', s=40, label='Original Point 2')

    axs[1].scatter(aug_point[0, 0], aug_point[0, 1], c='r', s=40,
                   label='Transformed Point 1', marker='x')
    axs[1].scatter(aug_point[1, 0], aug_point[1, 1], c='g',
                   s=40, label='Transformed Point 2', marker='x')

    plt.tight_layout()
    if plt.get_backend() != 'agg':
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

    if device == "cuda" and not torch.cuda.is_available():
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
        softbinarize_aug_prob=0.5,
        softbinarize_thr_quantile=0.01,
        softbinarize_blending_factor_minmax=(0.3, 0.7),
        is_torch_layout=False,
        min_max_brightness_factor=(1.0, 1.0),
        min_max_contrast_factor=(1.0, 1.0),
        brightness_aug_prob=brightness_aug_prob,
        contrast_aug_prob=contrast_aug_prob,
        input_normalization_factor=255.0,
        enable_auto_input_normalization=False,
        device=device
    )

    augs_helper = ImageAugmentationsHelper(cfg)

    device_t = torch.device(device)
    _assert_helper_modules_on_device(augs_helper, device_t)

    # Create dummy inputs on the target device
    x = torch.randn(1, 1, 64, 64, device=device)
    lbl = torch.tensor([[32, 32]], dtype=torch.float32, device=device)

    out_img, out_lbl = augs_helper(x, lbl)

    # Verify that outputs remain on the target device
    assert out_img.device.type == device
    assert out_lbl.device.type == device
    _assert_helper_modules_on_device(augs_helper, device_t)


def test_augmentation_helper_timing_cpu_vs_cuda():
    """Test execution time cpu vs cuda for augmentations helper"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    num_iters_ = 500
    cfg_kwargs = dict(
        max_shift_img_fraction=(0.2, 0.2),
        input_data_keys=[DataKey.IMAGE, DataKey.KEYPOINTS],
        shift_aug_prob=1.0,
        is_normalized=True,
        rotation_angle=(-10, 10),
        rotation_aug_prob=1.0,
        sigma_gaussian_noise_dn=0.2,
        gaussian_noise_aug_prob=1.0,
        gaussian_blur_aug_prob=1.0,
        softbinarize_aug_prob=1.0,
        softbinarize_thr_quantile=0.01,
        softbinarize_blending_factor_minmax=(0.3, 0.7),
        is_torch_layout=True,
        min_max_brightness_factor=(0.8, 1.2),
        min_max_contrast_factor=(0.8, 1.2),
        brightness_aug_prob=1.0,
        contrast_aug_prob=1.0,
        input_normalization_factor=1.0,
        enable_auto_input_normalization=False,
    )

    cpu_helper = ImageAugmentationsHelper(
        AugmentationConfig(**cfg_kwargs)).to("cpu")
    cuda_helper = ImageAugmentationsHelper(
        AugmentationConfig(**cfg_kwargs)).to("cuda")

    batch_size = 64
    x_cpu = torch.rand(batch_size, 1, 256, 256, device="cpu")
    lbl_cpu = torch.tensor(
        [[128, 128]], dtype=torch.float32).repeat(batch_size, 1)

    x_cuda = x_cpu.to("cuda")
    lbl_cuda = lbl_cpu.to("cuda")

    cpu_time_total, cpu_time_avg = _time_augmentation_helper(
        cpu_helper, x_cpu, lbl_cpu, num_iters=num_iters_)
    cuda_time_total, cuda_time_avg = _time_augmentation_helper(
        cuda_helper, x_cuda, lbl_cuda, num_iters=num_iters_)

    assert cpu_time_total > 0.0
    assert cuda_time_total > 0.0
    print(
        f"ImageAugmentationsHelper timing - cpu: {cpu_time_total:.4f}s, cuda: {cuda_time_total:.4f}s"
    )
    print(
        f"ImageAugmentationsHelper timing avg. per call - cpu: {cpu_time_avg:.4f}s, cuda avg: {cuda_time_avg:.4f}s")

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
    assert torch.allclose(
        out_img, x), "Output image should be the same as input"
    assert torch.allclose(
        out_lbl, lbl), "Output labels should be the same as input labels"


def test_transform_metadata_identity_when_no_geom_aug():
    cfg = AugmentationConfig(
        input_data_keys=[DataKey.IMAGE, DataKey.KEYPOINTS],
        shift_aug_prob=0.0,
        rotation_aug_prob=0.0,
        hflip_prob=0.0,
        vflip_prob=0.0,
        brightness_aug_prob=0.0,
        contrast_aug_prob=0.0,
        gaussian_blur_aug_prob=0.0,
        gaussian_noise_aug_prob=0.0,
        is_torch_layout=True,
        is_normalized=True,
        input_normalization_factor=1.0,
        enable_auto_input_normalization=False,
        device="cpu",
    )

    helper = ImageAugmentationsHelper(cfg)
    x = torch.rand(3, 1, 32, 32, dtype=torch.float32)
    lbl = torch.tensor([[10.0, 15.0], [12.0, 8.0], [4.0, 18.0]], dtype=torch.float32)

    (out_img, out_lbl), metadata = helper(x, lbl, return_transform_metadata=True)

    assert out_img.shape == x.shape
    assert out_lbl.shape == lbl.shape
    assert metadata is not None
    assert metadata.batch_size == x.shape[0]
    assert metadata.per_op_matrices_3x3 == {}
    assert metadata.geometric_ops_order == ()
    expected = torch.eye(3, dtype=out_img.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
    assert torch.allclose(metadata.combined_matrix_3x3.cpu(), expected, atol=1e-6)


def test_transform_metadata_affine_composition_and_getter():
    cfg = AugmentationConfig(
        input_data_keys=[DataKey.IMAGE, DataKey.KEYPOINTS],
        shift_aug_prob=1.0,
        max_shift_img_fraction=(0.2, 0.2),
        rotation_aug_prob=1.0,
        rotation_angle=(-30.0, 30.0),
        hflip_prob=1.0,
        vflip_prob=0.0,
        brightness_aug_prob=0.0,
        contrast_aug_prob=0.0,
        gaussian_blur_aug_prob=0.0,
        gaussian_noise_aug_prob=0.0,
        is_torch_layout=True,
        is_normalized=True,
        input_normalization_factor=1.0,
        enable_auto_input_normalization=False,
        device="cpu",
    )

    helper = ImageAugmentationsHelper(cfg)
    x = torch.rand(2, 1, 64, 64, dtype=torch.float32)
    lbl = torch.tensor([[20.0, 12.0], [40.0, 39.0]], dtype=torch.float32)

    (_, _), metadata = helper(x, lbl, return_transform_metadata=True)

    assert metadata is not None
    assert len(metadata.geometric_ops_order) >= 2
    assert metadata.combined_matrix_3x3.shape == (x.shape[0], 3, 3)

    composed = torch.eye(3, dtype=metadata.combined_matrix_3x3.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
    for op_key in metadata.geometric_ops_order:
        assert op_key in metadata.per_op_matrices_3x3
        op_matrix = metadata.per_op_matrices_3x3[op_key]
        assert op_matrix.shape == (x.shape[0], 3, 3)
        composed = op_matrix @ composed

    assert torch.allclose(metadata.combined_matrix_3x3, composed, atol=1e-5)

    cached_metadata = helper.Get_last_batch_transform_metadata()
    assert cached_metadata is not None
    assert torch.allclose(cached_metadata.combined_matrix_3x3, metadata.combined_matrix_3x3, atol=1e-6)

    cached_matrix = helper.Get_last_batch_transform_matrix_3x3()
    assert cached_matrix is not None
    assert torch.allclose(cached_matrix, metadata.combined_matrix_3x3, atol=1e-6)


def test_forward_default_signature_remains_compatible():
    cfg = AugmentationConfig(
        input_data_keys=[DataKey.IMAGE, DataKey.KEYPOINTS],
        shift_aug_prob=0.0,
        rotation_aug_prob=0.0,
        brightness_aug_prob=0.0,
        contrast_aug_prob=0.0,
        gaussian_blur_aug_prob=0.0,
        gaussian_noise_aug_prob=0.0,
        is_torch_layout=True,
        is_normalized=True,
        input_normalization_factor=1.0,
        enable_auto_input_normalization=False,
        device="cpu",
    )

    helper = ImageAugmentationsHelper(cfg)
    x = torch.rand(1, 1, 16, 16, dtype=torch.float32)
    lbl = torch.tensor([[5.0, 8.0]], dtype=torch.float32)

    out = helper(x, lbl)
    assert isinstance(out, (tuple, list))
    assert len(out) == 2
    assert torch.is_tensor(out[0])
    assert torch.is_tensor(out[1])


def test_update_clockwise_angle_identity_transform():
    input_clockwise_angle_from_positive_x_rad_batch = torch.tensor(
        [0.0, torch.pi / 2, torch.pi, 1.25], dtype=torch.float32
    )
    geometric_transform_matrix_batch_3x3 = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(4, 1, 1)

    updated_clockwise_angle_from_positive_x_rad_batch = (
        ImageAugmentationsHelper.Update_clockwise_angle_from_positive_x_using_geometric_transform(
            clockwise_angle_from_positive_x_rad_batch=input_clockwise_angle_from_positive_x_rad_batch,
            geometric_transform_matrix_batch=geometric_transform_matrix_batch_3x3,
            wrap_output_to_0_2pi=True,
        )
    )

    expected_clockwise_angle_from_positive_x_rad_batch = torch.remainder(
        input_clockwise_angle_from_positive_x_rad_batch, 2 * torch.pi
    )
    assert torch.allclose(
        updated_clockwise_angle_from_positive_x_rad_batch,
        expected_clockwise_angle_from_positive_x_rad_batch,
        atol=1e-6,
    )


def test_update_clockwise_angle_rotation_transform():
    rotation_delta_rad = torch.tensor(0.35, dtype=torch.float32)
    input_clockwise_angle_from_positive_x_rad_batch = torch.tensor(
        [0.25, 1.1, 2.4], dtype=torch.float32
    )

    cos_delta = torch.cos(rotation_delta_rad)
    sin_delta = torch.sin(rotation_delta_rad)
    geometric_transform_matrix_batch_3x3 = torch.tensor(
        [[
            [cos_delta.item(), -sin_delta.item(), 0.0],
            [sin_delta.item(), cos_delta.item(), 0.0],
            [0.0, 0.0, 1.0],
        ]],
        dtype=torch.float32,
    ).repeat(input_clockwise_angle_from_positive_x_rad_batch.shape[0], 1, 1)

    updated_clockwise_angle_from_positive_x_rad_batch = (
        ImageAugmentationsHelper.Update_clockwise_angle_from_positive_x_using_geometric_transform(
            clockwise_angle_from_positive_x_rad_batch=input_clockwise_angle_from_positive_x_rad_batch,
            geometric_transform_matrix_batch=geometric_transform_matrix_batch_3x3,
            wrap_output_to_0_2pi=True,
        )
    )

    expected_clockwise_angle_from_positive_x_rad_batch = torch.remainder(
        input_clockwise_angle_from_positive_x_rad_batch + rotation_delta_rad,
        2 * torch.pi,
    )
    assert torch.allclose(
        updated_clockwise_angle_from_positive_x_rad_batch,
        expected_clockwise_angle_from_positive_x_rad_batch,
        atol=1e-5,
    )


def test_update_clockwise_angle_horizontal_flip_transform():
    input_clockwise_angle_from_positive_x_rad_batch = torch.tensor(
        [0.2, 0.9, 2.7], dtype=torch.float32
    )
    geometric_transform_matrix_batch_3x3 = torch.tensor(
        [[[-1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    ).repeat(input_clockwise_angle_from_positive_x_rad_batch.shape[0], 1, 1)

    updated_clockwise_angle_from_positive_x_rad_batch = (
        ImageAugmentationsHelper.Update_clockwise_angle_from_positive_x_using_geometric_transform(
            clockwise_angle_from_positive_x_rad_batch=input_clockwise_angle_from_positive_x_rad_batch,
            geometric_transform_matrix_batch=geometric_transform_matrix_batch_3x3,
            wrap_output_to_0_2pi=True,
        )
    )

    expected_clockwise_angle_from_positive_x_rad_batch = torch.remainder(
        torch.pi - input_clockwise_angle_from_positive_x_rad_batch,
        2 * torch.pi,
    )
    assert torch.allclose(
        updated_clockwise_angle_from_positive_x_rad_batch,
        expected_clockwise_angle_from_positive_x_rad_batch,
        atol=1e-5,
    )


def test_update_clockwise_angle_degenerate_transform_preserves_input_and_supports_2x3():
    input_clockwise_angle_from_positive_x_rad_batch = torch.tensor(
        [[0.5], [1.4], [2.2]], dtype=torch.float32
    )
    geometric_transform_matrix_2x3 = torch.tensor(
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    updated_clockwise_angle_from_positive_x_rad_batch = (
        ImageAugmentationsHelper.Update_clockwise_angle_from_positive_x_using_geometric_transform(
            clockwise_angle_from_positive_x_rad_batch=input_clockwise_angle_from_positive_x_rad_batch,
            geometric_transform_matrix_batch=geometric_transform_matrix_2x3,
            wrap_output_to_0_2pi=False,
        )
    )

    assert updated_clockwise_angle_from_positive_x_rad_batch.shape == input_clockwise_angle_from_positive_x_rad_batch.shape
    assert torch.allclose(
        updated_clockwise_angle_from_positive_x_rad_batch,
        input_clockwise_angle_from_positive_x_rad_batch,
        atol=1e-6,
    )


def test_update_clockwise_angle_full_composed_affine_transform_matches_sequential_application():
    input_clockwise_angle_from_positive_x_rad_batch = torch.tensor(
        [0.15, 0.85, 1.65, 2.55], dtype=torch.float32
    )

    horizontal_flip_matrix_3x3 = torch.tensor(
        [[-1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    vertical_flip_matrix_3x3 = torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, -1.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    rotation_angle_rad = torch.tensor(0.43, dtype=torch.float32)
    rotation_matrix_3x3 = torch.tensor(
        [[torch.cos(rotation_angle_rad).item(), -torch.sin(rotation_angle_rad).item(), 0.0],
         [torch.sin(rotation_angle_rad).item(), torch.cos(rotation_angle_rad).item(), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    scale_shear_matrix_3x3 = torch.tensor(
        [[1.20, 0.18, 0.0],
         [-0.07, 0.92, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    translation_matrix_3x3 = torch.tensor(
        [[1.0, 0.0, 13.0],
         [0.0, 1.0, -9.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    composed_geometric_transform_matrix_3x3 = (
        translation_matrix_3x3
        @ scale_shear_matrix_3x3
        @ rotation_matrix_3x3
        @ vertical_flip_matrix_3x3
        @ horizontal_flip_matrix_3x3
    )

    updated_angles_from_composed_transform_rad_batch = (
        ImageAugmentationsHelper.Update_clockwise_angle_from_positive_x_using_geometric_transform(
            clockwise_angle_from_positive_x_rad_batch=input_clockwise_angle_from_positive_x_rad_batch,
            geometric_transform_matrix_batch=composed_geometric_transform_matrix_3x3,
            wrap_output_to_0_2pi=True,
        )
    )

    updated_angles_from_sequential_transforms_rad_batch = input_clockwise_angle_from_positive_x_rad_batch.clone()
    for geometric_transform_matrix_3x3 in (
        horizontal_flip_matrix_3x3,
        vertical_flip_matrix_3x3,
        rotation_matrix_3x3,
        scale_shear_matrix_3x3,
        translation_matrix_3x3,
    ):
        updated_angles_from_sequential_transforms_rad_batch = (
            ImageAugmentationsHelper.Update_clockwise_angle_from_positive_x_using_geometric_transform(
                clockwise_angle_from_positive_x_rad_batch=updated_angles_from_sequential_transforms_rad_batch,
                geometric_transform_matrix_batch=geometric_transform_matrix_3x3,
                wrap_output_to_0_2pi=True,
            )
        )

    assert torch.allclose(
        updated_angles_from_composed_transform_rad_batch,
        updated_angles_from_sequential_transforms_rad_batch,
        atol=1e-5,
    )


def test_update_clockwise_angle_different_transform_orders_produce_different_results():
    input_clockwise_angle_from_positive_x_rad_batch = torch.tensor(
        [0.35, 1.45, 2.75], dtype=torch.float32
    )

    horizontal_flip_matrix_3x3 = torch.tensor(
        [[-1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    rotation_angle_rad = torch.tensor(0.61, dtype=torch.float32)
    rotation_matrix_3x3 = torch.tensor(
        [[torch.cos(rotation_angle_rad).item(), -torch.sin(rotation_angle_rad).item(), 0.0],
         [torch.sin(rotation_angle_rad).item(), torch.cos(rotation_angle_rad).item(), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    composed_transform_rotate_then_flip_3x3 = horizontal_flip_matrix_3x3 @ rotation_matrix_3x3
    composed_transform_flip_then_rotate_3x3 = rotation_matrix_3x3 @ horizontal_flip_matrix_3x3

    updated_angles_rotate_then_flip_rad_batch = (
        ImageAugmentationsHelper.Update_clockwise_angle_from_positive_x_using_geometric_transform(
            clockwise_angle_from_positive_x_rad_batch=input_clockwise_angle_from_positive_x_rad_batch,
            geometric_transform_matrix_batch=composed_transform_rotate_then_flip_3x3,
            wrap_output_to_0_2pi=True,
        )
    )
    updated_angles_flip_then_rotate_rad_batch = (
        ImageAugmentationsHelper.Update_clockwise_angle_from_positive_x_using_geometric_transform(
            clockwise_angle_from_positive_x_rad_batch=input_clockwise_angle_from_positive_x_rad_batch,
            geometric_transform_matrix_batch=composed_transform_flip_then_rotate_3x3,
            wrap_output_to_0_2pi=True,
        )
    )

    assert not torch.allclose(
        updated_angles_rotate_then_flip_rad_batch,
        updated_angles_flip_then_rotate_rad_batch,
        atol=1e-6,
    )


def test_helper_metadata_supports_external_sun_direction_label_update():
    torch.manual_seed(11)

    batch_size = 4
    input_image_batch = torch.rand(batch_size, 1, 48, 48, dtype=torch.float32)
    input_label_batch = torch.tensor(
        [
            [24.0, 20.0, 45.0, 0.25],
            [12.0, 10.0, 60.0, 1.00],
            [36.0, 30.0, 90.0, 2.20],
            [18.0, 28.0, 15.0, 5.60],
        ],
        dtype=torch.float32,
    )

    augmentation_config = AugmentationConfig(
        input_data_keys=[DataKey.IMAGE, DataKey.KEYPOINTS],
        is_torch_layout=True,
        is_normalized=True,
        input_normalization_factor=1.0,
        enable_auto_input_normalization=False,
        # deterministic geometric transform for expected-angle check
        hflip_prob=1.0,
        vflip_prob=0.0,
        shift_aug_prob=0.0,
        rotation_aug_prob=0.0,
        brightness_aug_prob=0.0,
        contrast_aug_prob=0.0,
        gaussian_blur_aug_prob=0.0,
        gaussian_noise_aug_prob=0.0,
        device="cpu",
    )

    augmentation_helper = ImageAugmentationsHelper(augmentation_config)
    augmented_image_batch, augmented_label_batch = augmentation_helper(
        input_image_batch, input_label_batch
    )

    batch_geometric_transform_metadata = augmentation_helper.Get_last_batch_transform_metadata()
    assert batch_geometric_transform_metadata is not None
    assert batch_geometric_transform_metadata.combined_matrix_3x3.shape == (batch_size, 3, 3)

    # The helper updates only KEYPOINTS (x, y); extra label entries remain unchanged.
    assert torch.allclose(augmented_label_batch[:, 3], input_label_batch[:, 3], atol=1e-6)

    updated_sun_direction_angle_from_positive_x_rad_batch = (
        ImageAugmentationsHelper.Update_clockwise_angle_from_positive_x_using_geometric_transform(
            clockwise_angle_from_positive_x_rad_batch=input_label_batch[:, 3],
            geometric_transform_matrix_batch=batch_geometric_transform_metadata.combined_matrix_3x3,
            wrap_output_to_0_2pi=True,
        )
    )

    expected_updated_sun_direction_angle_from_positive_x_rad_batch = torch.remainder(
        torch.pi - input_label_batch[:, 3],
        2 * torch.pi,
    )

    assert torch.allclose(
        updated_sun_direction_angle_from_positive_x_rad_batch,
        expected_updated_sun_direction_angle_from_positive_x_rad_batch,
        atol=1e-5,
    )

    # This is the angle update that external training code should apply.
    externally_updated_label_batch = augmented_label_batch.clone()
    externally_updated_label_batch[:, 3] = updated_sun_direction_angle_from_positive_x_rad_batch
    assert not torch.allclose(externally_updated_label_batch[:, 3], input_label_batch[:, 3], atol=1e-6)


# %% MANUAL TEST CALLS
if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shift_aug_prob = 0.0
    rotation_aug_prob = 0.0
    gaussian_noise_aug_prob = 0.0
    gaussian_blur_aug_prob = 0.0
    brightness_aug_prob = 0.0
    contrast_aug_prob = 0.0

    # test_synthetic_mask_augmentation()
    # test_sample_images_augmentation()
    # test_random_softbinarize_only_sample_images_visualization()
    # test_AugmentationSequential()
    #test_augmentation_helper_preserves_device(device,
    #                                        shift_aug_prob,
    #                                        rotation_aug_prob,
    #                                        gaussian_noise_aug_prob,
    #                                        gaussian_blur_aug_prob,
    #                                        brightness_aug_prob,
    #                                        contrast_aug_prob)
    # test_zero_augs_input_unchanged()
    # test_random_gaussian_noise_validation_filters_dark_images()
    # test_detect_border_crossing_white_masks()
    test_augmentation_helper_timing_cpu_vs_cuda()
