# Script implementing a classification example on CIFAR100 as exercise by PeterC, using pretrained models in torchvision - 23-07-2024
# Reference: https://pytorch.org/vision/0.9/models.html#classification

# Import modules
import torch
from torch import nn
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
from torchvision import transforms
import mlflow
import numpy as np

import sys, os
import pyTorchAutoForge
import torchvision.models as models

wsDir = '/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/tutorials'
if os.curdir != wsDir:
    os.chdir(wsDir)

def main():
    model = models.resnet18(weights=None) 
    # NOTE: all models in torchvision.models for classification are trained on ImageNet, thus have 1000 classes as output
    print(model)

    # Therefore, let's modify the last layer! (Named fc)
    numInputFeatures = model.fc.in_features # Same as what the model uses 
    numOutClasses = 10 # Selected by user 

    model.fc = nn.Linear(in_features=numInputFeatures,
                         out_features=numOutClasses, bias=True)  # 2 classes in our case
    
    print(model) # Check last layer now

    # %% Load CIFAR10 dataset from torchvision
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Apply normalization

    # Get dataset objects
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    validation_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

    # %% Define loss function and optimizer
    initial_lr = 1E-4   

    lossFcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, fused=True)

    trainerConfig = ModelTrainingManager.ModelTrainingManagerConfig(
        initial_lr=initial_lr, lr_scheduler=None)

    dataloaderIndex = pyTorchAutoForge.dataloaderIndex(train_loader, validation_loader)

    # Perform training and validation of model
    pyTorchAutoForge.TrainAndValidateModel(dataloaderIndex, model, lossFcn, optimizer, options)
    
    # TODO: Update customTorchTool to support training of models for classification

    with mlflow.start_run() as mlflow_run:
        print('Start run: {name}'.format(name=mlflow_run.info.run_name))


        mlflow.end_run()



if __name__ == '__main__':
    mlflow.set_experiment('ImageClassificationExample_CIFAR100')
    main()