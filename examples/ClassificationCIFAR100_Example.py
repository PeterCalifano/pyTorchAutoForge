# Script implementing a classification example on CIFAR100 as exercise by PeterC, using pretrained models in torchvision - 19-09-2024
# Reference: https://pytorch.org/vision/0.9/models.html#classification

# Import modules
import torch
from torch import nn
# Utils for dataset management, storing pairs of (sample, label)
from torch.utils.data import DataLoader
from torchvision import datasets  # Import vision default datasets from torchvision
from torchvision import transforms
import mlflow, copy

from pyTorchAutoForge.optimization.ModelTrainingManager import ModelTrainingManager, ModelTrainingManagerConfig, TaskType, TrainModel, ValidateModel
from pyTorchAutoForge.datasets import DataloaderIndex
import torchvision.models as models

from pyTorchAutoForge.utils import GetDevice

def main():

    # NOTE: seems that mlflow is not using server location

    # Set mlflow experiment
    mlflow.set_experiment('ImageClassificationCIFAR100_Example_TEST')

    # Get model backbone from torchvision
    # All models in torchvision.models for classification are trained on ImageNet, thus have 1000 classes as output
    model = models.resnet18(weights=False)  # Load pretrained weights
    print(model)

    device = GetDevice()

    # Therefore, let's modify the last layer! (Named fc)
    numInputFeatures = model.fc.in_features  # Same as what the model uses
    numOutClasses = 10  # Selected by user
    model.fc = (nn.Linear(in_features=numInputFeatures,
                          out_features=numOutClasses, bias=True))  # 100 classes in our case
    model.to(device)
    print(model)  # Check last layer now

    # %% Load CIFAR100 dataset from torchvision
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Apply normalization

    # Get dataset objects
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    validation_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    # Define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=64, shuffle=False)

    # %% Define loss function and optimizer
    initial_lr = 1E-4
    numOfEpochs = 20
    lossFcn = nn.CrossEntropyLoss()

    fused = True if device == "cuda:0" else False
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, fused=fused)

    # Define model training manager config  (dataclass init)
    trainerConfig = ModelTrainingManagerConfig(tasktype=TaskType.CLASSIFICATION,
        initial_lr=initial_lr, lr_scheduler=None, num_of_epochs=numOfEpochs, optimizer=optimizer)
    
    # DEVNOTE: TODO, add check on optimizer from ModelTrainingManagerConfig. It must be of optimizer type
    print("\nModelTrainingManagerConfig instance:", trainerConfig)

    print("\nDict of ModelTrainingManagerConfig instance:", trainerConfig.getConfigDict())
    
    # Define model training manager instance
    trainer = ModelTrainingManager(model=model, lossFcn=lossFcn, config=trainerConfig, optimizer=0)
    print("\nModelTrainingManager instance:", trainer)

    # Define dataloader index for training
    dataloaderIndex = DataloaderIndex(train_loader, validation_loader)
    trainer.setDataloaders(dataloaderIndex) # Set dataloaders for training and validation

    # Perform training and validation of model

    # CHECK: versus TrainAndValidateModel
    model2 = copy.deepcopy(model).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=initial_lr, fused=fused)

    for epoch in range(numOfEpochs):
        print(f"Epoch TEST: {epoch+1}/{numOfEpochs}")
        TrainModel(dataloaderIndex.getTrainLoader(), model2, lossFcn, optimizer2, 0)
        ValidateModel(dataloaderIndex.getValidationLoader(), model2, lossFcn)

    trainer.trainAndValidate()

    for param1, param2 in zip(model.parameters(), trainer.model.parameters()):
        if not(torch.equal(param1, param2)) or not(param1 is param2):
            raise ValueError("Model parameters are not the same. optimizer does not update trainer.model!")



if __name__ == '__main__':
    main()
