# Script implementing the PyTorch "Dataset and Dataloaders" example as exercise by PeterC - 28-05-2024
# Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

# Built-in PyTorch utilities to manage datasets: 
# 1) torch.utils.data.Dataloader --> wraps Dataset content as iterable for easy access in loops
# 2) torch.utils.data.Dataset --> stores the samplers and the corresponding labels 

# Default datasets have the following inputs:
# root:      string indicating path where data are stored
# training:  bool specifying whether data are for training or testing
# download:  bool specifying whether to download data
# transform: function specifying any operation to perform on the dataset when loaded.

# %% EXAMPLE: MNIST

import torch
import os 
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image

# Get training data and testing data lists
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Plot samples 
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# %% CUSTOM DATASET

class CustomImageDataset(Dataset):
    # __init__ function: called one to instantiate the Dataset object
    # For instance, it specifies which are the labels, where the images are (or the inputs), the transformation to apply to the data,
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # __len__ returns number of samples in the dataset
    def __len__(self):
        return len(self.img_labels)

    # __getitem__ extracts the ith pair (input, label) at the given idx
    # In this case it identifies the image number and loads it in memory using read_image, then calls the transform function (e.g. for augmentation). 
    # The latter step is anyway required as default to produce a PyTorch tensor as output
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# The advantage of using Dataset is the automatic batching and shuffling Dataloader enables (built-in)
# Dataloader directly extracts from the entire dataset in Dataset object training and testing samples in mini batches in a random way.
# Ordering and random component of the extraction can be fine-tuned controlling the "Samplers" object.
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

