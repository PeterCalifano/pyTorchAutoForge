from __future__ import annotations
from pyTorchAutoForge.datasets.DatasetClasses import PTAF_Datakey, FetchDatasetPaths, DatasetPathsContainer

from pyTorchAutoForge.datasets.LabelsClasses import LabelsContainer

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING, Type
import torch
from torchvision.transforms import ToTensor
import numpy as np
import gc

from pyTorchAutoForge.datasets.DatasetClasses import PTAF_Datakey, ImagesLabelsContainer

try:
    from cv2 import imread
    def LoadDatasetToMem(dataset_name: str | list | tuple,
                        datasets_root_folder: str | tuple[str, ...],
                        dataset_origin_tag: tuple[Literal['legacy',
                                                        'dataset_gen_lib', 'dataset_gen_lib_abram']],
                        lbl_vector_data_keys: tuple[PTAF_Datakey | str, ...],
                        samples_limit_per_dataset: int = 0,
                        img_width_heigth: int | tuple[int, int] = (1024, 1024),
                        label_vector_size: int | None = None,
                        max_apparent_size: float | int | None = None,
                        min_bbox_width_height: tuple[float, float] | float | None = None) -> ImagesLabelsContainer:
        """
        Loads a dataset of images and labels into memory as tensors.

        Args:
            dataset_name (str | list | tuple): Name(s) of the dataset(s) to load.
            datasets_root_folder (str | tuple[str, ...]): Root folder(s) where datasets are stored.
            dataset_origin_tag (tuple[Literal['legacy', 'dataset_gen_lib', 'dataset_gen_lib_abram']]): Tag indicating the dataset's origin.
            lbl_vector_data_keys (tuple[PTAF_Datakey | str, ...]): Keys specifying which label data to extract.
            samples_limit_per_dataset (int, optional): Maximum number of samples to load per dataset. Defaults to 0 (no limit).
            img_width_heigth (int | tuple[int, int], optional): Image width and height. Defaults to (1024, 1024).
            label_vector_size (int | None, optional): Size of the label vector. If None, inferred from data. Defaults to None.

        Returns:
            ImagesLabelsContainer: Container with loaded images and labels.

        Raises:
            TypeError: If input types are incorrect.
            ValueError: If label file format is unsupported or other value errors occur.

        """

        img_width, img_height = img_width_heigth if isinstance(
            img_width_heigth, (list, tuple)) else (img_width_heigth, img_width_heigth)

        # Select loading mode (single or multiple datasets)
        if isinstance(dataset_name, str):
            dataset_names_array: list | tuple = (dataset_name,)

        elif isinstance(dataset_name, (list, tuple)):
            dataset_names_array: list | tuple = dataset_name
        else:
            raise TypeError(
                "dataset_name must be a string or a list of strings")

        rejected_data_index = []

        # Fetch dataset paths
        dataset_paths_container = FetchDatasetPaths(dataset_name=dataset_names_array,
                                                    datasets_root_folder=datasets_root_folder,
                                                    samples_limit_per_dataset=samples_limit_per_dataset)

        imgPaths, lblPaths, total_num_imgs = dataset_paths_container.dump_as_tuple()

        if total_num_imgs == 0:
            raise ValueError("No valid images found in dataset paths container. Something may have gone wrong.")

        # Allocate tensors for images and labels
        imgData = torch.zeros(total_num_imgs, img_height, img_width, dtype=torch.uint8)
        lblData = None
        if label_vector_size is not None:
            # Initialize label data tensor with specified size
            lblData = torch.zeros(total_num_imgs, label_vector_size, dtype=torch.float32)

        # Start loading process
        toTensor = ToTensor()
        
        current_dtype = ""

        for id, (imgPath, labelPath) in enumerate(zip(imgPaths, lblPaths)):

            scale_bit_depth = False
            if dataset_origin_tag == 'dataset_gen_lib_abram':
                scale_bit_depth = True

            # Load image
            tmpImage = imread(imgPath, -1)

            # Check the data type
            image_scaling_coeff = 1.0

            if tmpImage.dtype == 'uint8' and not scale_bit_depth:
                image_scaling_coeff = 1.0
                if current_dtype != tmpImage.dtype:
                    print("\nLoading uint8 (8-bit) images...")
                    current_dtype = tmpImage.dtype

                # Set all 1.0 to 0.0 to fix Blender images issue
                tmpImage[tmpImage == 1.0] = 0.0

            elif tmpImage.dtype == 'uint16' and not scale_bit_depth:
                image_scaling_coeff = 255.0 / 65535.0
                if current_dtype != tmpImage.dtype:
                    print("\nLoading uint16 (16-bit) images...")
                    current_dtype = tmpImage.dtype

                # Set all 1.0 to 0.0 to fix Blender images issue
                tmpImage[tmpImage == 1.0] = 0.0

            elif tmpImage.dtype == 'uint8' and scale_bit_depth:
                # Images from ABRAM have 12-bit depth?
                image_scaling_coeff = 1.0
                if current_dtype != tmpImage.dtype:
                    print("\nLoading uint8 (8-bit) images (dataset_gen_lib_abram)...")
                    current_dtype = tmpImage.dtype

            elif tmpImage.dtype != 'uint8' and tmpImage.dtype != 'uint16':
                raise TypeError(
                    "Image data type is neither uint8 nor uint16. This dataset loader only supports these two data types.")

            # Set all entries <= 7 to 0.0
            scaled_img = image_scaling_coeff * tmpImage.astype(np.float32)
            pix_zero_mask = scaled_img <= 7
            scaled_img[pix_zero_mask] = 0.0
            print(f"\rLoading image {id+1}/{total_num_imgs}", end='')

            # Check if image median intensity is > 30
            nonzero_pixels = scaled_img[scaled_img > 0]
            perc75_intensity = np.percentile(
                nonzero_pixels, 75) if nonzero_pixels.size > 0 else 0

            if perc75_intensity < 5 or (perc75_intensity < 10 and sum(nonzero_pixels) < 2E3) or (perc75_intensity < 7 and sum(nonzero_pixels) < 5E3):
                print(
                    f" - Warning: Image {id+1} has 75th percentile intensity of illuminated pixels equal to {perc75_intensity}, discarded as too dark.")
                rejected_data_index.append(id)
                continue

            print(f"\rLoading image {id+1}/{total_num_imgs}", end='', flush=True)
            tmpImage[pix_zero_mask] = 0.0

            imgData[id, :, :] = toTensor(
                np.round(image_scaling_coeff * tmpImage.astype(np.float32))
                )

            # Load label
            # DEPRECATED CODE to remove
            # if labelPath.endswith('.json'):
            #    with open(labelPath) as file:
            #        # If label is .json file, load it
            #        labelFile = json.load(file)
            #        lblData[id, 0:2] = torch.tensor(
            #            labelFile["dCentroid"], dtype=torch.float32)
            #        lblData[id, 2] = torch.tensor(
            #            labelFile["dRangeInRadii"], dtype=torch.float32)
            #        # lblData[id, 3] = torch.tensor(
            #        #    labelFile["dRadiusInPix"], dtype=torch.float32)

            if labelPath.endswith('.yml') or labelPath.endswith('.yaml'):
                # Load yml file
                labelFile = LabelsContainer.load_from_yaml(labelPath)

                if label_vector_size is None:
                    # Get size from container object
                    label_vector_size, sizes_dict = LabelsContainer.get_lbl_1d_vector_size(
                        data_keys=lbl_vector_data_keys)
                else:
                    sizes_dict = {}

                if lblData is None:
                    # Initialize label data tensor
                    lblData = torch.zeros(total_num_imgs, label_vector_size, dtype=torch.float32)

                # Get labels corresponding to datakeys
                label_values = labelFile.get_labels(data_keys=lbl_vector_data_keys)
                lblData[id, :] = torch.tensor(label_values, dtype=torch.float32)

                # If discard based on apparent size is requested check it
                if max_apparent_size is not None:
                    lbl_check_val = labelFile.get_labels(data_keys=PTAF_Datakey.APPARENT_SIZE)

                    if lbl_check_val is not None and lbl_check_val > max_apparent_size:
                        print(f" - Warning: Image {id+1} has apparent size {lbl_check_val}, discarded as too large.")
                        rejected_data_index.append(id)
                        continue    

                # If discard based on bounding box size is requested check it
                if min_bbox_width_height is not None:
                    if isinstance(min_bbox_width_height, (float, int)):
                        min_bbox_width_height = (min_bbox_width_height, min_bbox_width_height)

                    lbl_check_val = labelFile.get_labels(data_keys=PTAF_Datakey.BBOX_XYWH)

                    if lbl_check_val is not None and lbl_check_val[2] < min_bbox_width_height[0] and lbl_check_val[3] < min_bbox_width_height[1]:
                        print(f" - Warning: Image {id+1} has bounding box width, height = {lbl_check_val[2:4]} < {min_bbox_width_height}, discarded as too small.")
                        rejected_data_index.append(id)
                        continue

            else:
                raise ValueError(
                    f"Unsupported label file format: {labelPath}. Only .yml and .yaml are supported.")

        print("\n")
        # Remove indices in rejected_data_index
        if len(rejected_data_index) > 0:

            mask = torch.ones(total_num_imgs,
                            dtype=torch.bool,
                            device=imgData.device)

            mask[rejected_data_index] = False

            imgData = imgData[mask]
            lblData = lblData[mask]

        # Check input types before assigning to ImagesLabelsContainer
        if not isinstance(imgData, torch.Tensor):
            raise TypeError(
                f"Expected imgData to be a torch.Tensor, got {type(imgData)}")

        if not isinstance(lblData, torch.Tensor):
            raise TypeError(
                f"Expected lblData to be a torch.Tensor, got {type(lblData)}")

        print(
            f"Rejected {len(rejected_data_index)}/{total_num_imgs} images due to low median intensity.")

        # Pack output
        return ImagesLabelsContainer(images=imgData,
                                    labels=lblData,
                                    labels_datakeys=lbl_vector_data_keys,
                                    labels_sizes=sizes_dict)
except ImportError:
    print("\033[91mOpenCV is not installed. LoadDatasetToMem function will not work.\033[0m")
    def LoadDatasetToMem(*args, **kwargs):
        raise ImportError("OpenCV is required for LoadDatasetToMem function.") #type:ignore