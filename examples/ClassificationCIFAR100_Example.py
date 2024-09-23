# Script implementing a classification example on CIFAR100 as exercise by PeterC, using pretrained models in torchvision - 19-09-2024
# Reference: https://pytorch.org/vision/0.9/models.html#classification

# Import modules
import torch
from torch import nn
# Utils for dataset management, storing pairs of (sample, label)
from torch.utils.data import DataLoader
from torchvision import datasets  # Import vision default datasets from torchvision
from torchvision import transforms
import mlflow

from pyTorchAutoForge.optimization.ModelTrainingManager import ModelTrainingManager, ModelTrainingManagerConfig
from pyTorchAutoForge.datasets import DataloaderIndex
import pyTorchAutoForge.optimization as optim
import torchvision.models as models

from pyTorchAutoForge.utils import GetDevice


def main():

    # NOTE: seems that mlflow is not using server location

    # Set mlflow experiment
    mlflow.set_experiment('ImageClassificationCIFAR100_Example')

    # Get model backbone from torchvision
    # All models in torchvision.models for classification are trained on ImageNet, thus have 1000 classes as output
    model = models.resnet18(weights=None)
    print(model)

    device = GetDevice()

    # Therefore, let's modify the last layer! (Named fc)
    numInputFeatures = model.fc.in_features  # Same as what the model uses
    numOutClasses = 100  # Selected by user
    model.fc = (nn.Linear(in_features=numInputFeatures,
                          out_features=numOutClasses, bias=True))  # 100 classes in our case
    model.to(device)
    print(model)  # Check last layer now

    # %% Load CIFAR100 dataset from torchvision
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Apply normalization

    # Get dataset objects
    train_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform)
    validation_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform)

    # Define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=64, shuffle=False)

    # %% Define loss function and optimizer
    initial_lr = 1E-4

    lossFcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, fused=True)

    # Define model training manager config  (dataclass init)
    trainerConfig = ModelTrainingManagerConfig(
        initial_lr=initial_lr, lr_scheduler=None, optimizer=optimizer)
    print("\nModelTrainingManagerConfig instance:", trainerConfig)

    print("\nDict of ModelTrainingManagerConfig instance:", trainerConfig.getConfigDict())

    # Define model training manager instance
    trainer = ModelTrainingManager(model=model, lossFcn=lossFcn, config=trainerConfig)
    print("\nModelTrainingManager instance:", trainer)

    # Define dataloader index for training
    dataloaderIndex = DataloaderIndex(train_loader, validation_loader)

    # Perform training and validation of model
    optim.TrainAndValidateModel(dataloaderIndex, model, lossFcn, optimizer)

    # TODO: Update customTorchTool to support training of models for classification

    with mlflow.start_run() as mlflow_run:
        print('Start run: {name}'.format(name=mlflow_run.info.run_name))
        mlflow.end_run()


if __name__ == '__main__':
    main()
