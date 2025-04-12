import enum
from re import L
from torch.utils.data import Dataset
import numpy as np
import torch
from dataclasses import dataclass

from zipp import Path
from pyTorchAutoForge.utils import numpy_to_torch

# %% EXPERIMENTAL: Generic Dataset class for Supervised learning - 30-05-2024
# Base class for Supervised learning datasets
# Reference for implementation of virtual methods: https://stackoverflow.com/questions/4714136/how-to-implement-virtual-methods-in-python
from abc import abstractmethod
from abc import ABCMeta

class DatasetScope(enum.Enum):
    """
    DatasetScope class to define the scope of a dataset.
    Attributes:
        TRAINING (str): Represents the training dataset.
        TEST (str): Represents the test dataset.
        VALIDATION (str): Represents the validation dataset.
    """
    TRAINING = 'train'
    TEST = 'test'
    VALIDATION = 'validation'

    def __str__(self):
        return self.value
    def __repr__(self):
        return self.value
    def __eq__(self, other):
        if isinstance(other, DatasetScope):
            return self.value == other.value


@dataclass
class ImagesLabelsContainer:
    """
     _summary_

    _extended_summary_
    """
    images : np.ndarray | torch.Tensor
    labels : np.ndarray | torch.Tensor
    
    
class ImagesLabelsDataset(Dataset):
    """
    ImagesLabelsDataset _summary_

    _extended_summary_

    :param Dataset: _description_
    :type Dataset: _type_
    """

    def __init__(self, images_labels: ImagesLabelsContainer | None = None, # type: ignore
                 transforms: None = None, # TODO which type?
                 images_path: str | None = None, 
                 labels_path: str | None = None) -> None:
        
        # Store input and labels sources
        if images_labels is None and (images_path is None or labels_path is None):
            raise ValueError("Either images_labels container or both images_path and labels_path must be provided.")
        
        elif not (images_path is None or labels_path is None):
            # Load dataset from paths
            images_labels: ImagesLabelsContainer = self.load_from_paths(images_path, labels_path)

        if images_labels is None:
            raise ValueError("images_labels container is None after loading from paths. Something may have gone wrong. Report this issue please.")
        
        self.images = numpy_to_torch(images_labels.images)
        self.labels = numpy_to_torch(images_labels.labels)

        # Initialize transform objects
        self.transforms = transforms

    def __len__(self):
        # Number of images is 4th dimension of the torch Tensor
        return np.shape(self.images)[3]


    # TODO investigate difference between __getitem__ and __getitems__
    def __getitem__(self, index):
        # Get data
        image = self.images[index, :, :, :]
        label = self.labels[index, :]

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label

    # Batch fetching
    def __getitems__(self, list_of_indices):

        # Create a numpy array from the list of indices
        indices = np.array(list_of_indices)

        # Get data
        image = self.images[indices, :, :, :]
        label = self.labels[indices, :]

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label
    
    def load_from_paths(self, images_path:str, labels_path:str) -> ImagesLabelsContainer:
        images, labels = [], [] # TODO
        return ImagesLabelsContainer(images, labels)


# TODO function to rework as method of ImagesLabelsDataset
def LoadDataset(datasetID: Union[int, list[int]], datasetsRootFolder: str, hostname: str, limit: int = 0) -> dict:

    # Select loading mode (single or multiple datasets)
    if isinstance(datasetID, int):
        datasetID_array = [datasetID]
    elif isinstance(datasetID, list):
        datasetID_array = datasetID
    else:
        raise TypeError("datasetID must be an integer or a list of integers")

    # Get index list of datasets and print
    with open(datasetsRootFolder + "/datasetList.json") as datasetListFile:
        fileDict = json.load(datasetListFile)
        localDatasetNames = [fileDict["datasetFolders"][id] for id in range(
            len(fileDict["datasetFolders"])) if fileDict["hostDeviceName"][id] == hostname]

        print("Available datasets: ", localDatasetNames)

    # Initialize index of datasets to load
    image_folder = []
    label_folder = []
    numOfImagesInSets = []

    imgPaths = []
    lblPaths = []

    for count, datasetID in enumerate(datasetID_array):

        # Get dataset paths
        image_folder.append(fileDict["datasetPaths"]["images"][datasetID])
        label_folder.append(fileDict["datasetPaths"]["labels"][datasetID])

        # Load all images into torch.tensor
        numOfImagesInSets.append(len(os.listdir(image_folder[count])))

        # Check size of names in the folder
        sample_file = next((f for f in os.listdir(image_folder[count]) if os.path.isfile(
            os.path.join(image_folder[count], f))), None)

        if sample_file:
            name_size = len(os.path.splitext(sample_file)[0])
            print(f"Name size is: {name_size}")
        else:
            print("No files found in the folder.")

        # Build paths index

        if name_size == 6:
            imgPaths.extend([os.path.join(
                image_folder[count], f"{id+1:06d}.png") for id in range(numOfImagesInSets[count])])
        elif name_size == 8:
            imgPaths.extend([os.path.join(
                image_folder[count], f"{id*150:08d}.png") for id in range(numOfImagesInSets[count])])

        lblPaths.extend([os.path.join(
            label_folder[count], f"{id+1:06d}.json") for id in range(numOfImagesInSets[count])])

    totalNumOfImages = sum(numOfImagesInSets)

    if limit > 0:
        totalNumOfImages = min(totalNumOfImages, limit)
        imgPaths = imgPaths[:limit]
        lblPaths = lblPaths[:limit]

    # Allocate tensors for images and labels
    imgData = torch.zeros(1, 1024, 1024, totalNumOfImages, dtype=torch.uint8)
    lblData = torch.zeros(4, totalNumOfImages, dtype=torch.float32)

    for id, (imgPath, labelPath) in enumerate(zip(imgPaths, lblPaths)):

        # Load image
        tmpImage = ocv.imread(imgPath, -1)

        # Check the data type
        if tmpImage.dtype == 'uint8' and id == 0:
            imageScalingValue = 1.0
            print("\nLoading uint8 (8-bit) images...\n")
        elif tmpImage.dtype == 'uint16' and id == 0:
            imageScalingValue = 1.0/256.0
            print("\nLoading uint16 (16-bit) images...\n")
        elif tmpImage.dtype != 'uint8' and tmpImage.dtype != 'uint16' and id == 0:
            raise TypeError("Image data type is not uint8 or uint16.")

        print(f"\rLoading image {id+1}/{totalNumOfImages}", end='', flush=True)
        imgData[0, :, :, id] = torch.tensor(
            imageScalingValue * tmpImage, dtype=torch.uint8).permute(0, 1)

        # Load label
        labelFile = json.load(open(labelPath))
        lblData[0:2, id] = torch.tensor(
            labelFile["dCentroid"], dtype=torch.float32)
        lblData[2, id] = torch.tensor(
            labelFile["dRangeInRadii"], dtype=torch.float32)
        lblData[3, id] = torch.tensor(
            labelFile["dRadiusInPix"], dtype=torch.float32)
    print("\n")

    # Create dictionary
    dataDict = {'images': imgData, 'labels': lblData}

    return dataDict

# %% EXPERIMENTAL STUFF
# TODO: python Generics to implement? EXPERIMENTAL
class GenericSupervisedDataset(Dataset, metaclass=ABCMeta):
    """
    A generic dataset class for supervised learning.

    This class serves as a base class for supervised learning datasets. It 
    provides a structure for handling input data, labels, and dataset types 
    (e.g., training, testing, validation). Subclasses must implement the 
    abstract methods to define specific dataset behavior.

    Args:
        input_datapath (str): Path to the input data.
        labels_datapath (str): Path to the labels data.
        dataset_type (str): Type of the dataset (e.g., 'train', 'test', 'validation').
        transform (callable, optional): A function/transform to apply to the input data. Defaults to None.
        target_transform (callable, optional): A function/transform to apply to the target labels. Defaults to None.
    """

    def __init__(self, input_datapath: str, labels_datapath: str,
                 dataset_type: str, transform=None, target_transform=None):

        # Store input and labels sources
        self.labels_dir = labels_datapath
        self.input_dir = input_datapath

        # Initialize transform objects
        self.transform = transform
        self.target_transform = target_transform

        # Set the dataset type (train, test, validation)
        self.dataset_type = dataset_type

    def __len__(self):
        return len()  # TODO

    @abstractmethod
    def __getLabelsData__(self):
        raise NotImplementedError()
        # Get and store labels vector
        self.labels  # TODO: "Read file" of some kind goes here. Best current option: write to JSON

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()
        return inputVec, label
