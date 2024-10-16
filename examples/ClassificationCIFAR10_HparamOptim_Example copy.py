# Script implementing a classification example on CIFAR10 as exercise by PeterC, using pretrained models in torchvision - 19-09-2024
# Reference: https://pytorch.org/vision/0.9/models.html#classification

# Import modules
import torch
from torch import nn
# Utils for dataset management, storing pairs of (sample, label)
from torch.utils.data import DataLoader
from torchvision import datasets  # Import vision default datasets from torchvision
from torchvision import transforms
import mlflow, optuna

from pyTorchAutoForge.optimization.ModelTrainingManager import ModelTrainingManager, ModelTrainingManagerConfig, TaskType, TrainModel, ValidateModel
from pyTorchAutoForge.datasets import DataloaderIndex
import torchvision.models as models

from pyTorchAutoForge.utils import GetDevice
from ClassificationCIFAR10_Example import DefineDataloaders, DefineModel, DefineOptimStrategy


def main():

    # Set mlflow experiment
    mlflow.set_experiment('ImageClassificationCIFAR10_HparamOptim_Example')
    
    # Get model backbone from torchvision
    # All models in torchvision.models for classification are trained on ImageNet, thus have 1000 classes as output
    model = DefineModel()
    device = GetDevice()
    model.to(device)
    print(model)  # Check last layer

    # %% Define loss function and optimizer
    lossFcn, initial_lr = DefineOptimStrategy()
    numOfEpochs = 50

    fused = True if device == "cuda:0" else False
    optimizer = torch.optim.Adam(
        model.parameters(), lr=initial_lr, fused=fused)

    # Define model training manager config  (dataclass init)
    trainerConfig = ModelTrainingManagerConfig(tasktype=TaskType.CLASSIFICATION,
                                               initial_lr=initial_lr, lr_scheduler=None, num_of_epochs=numOfEpochs, optimizer=optimizer)

    # DEVNOTE: TODO, add check on optimizer from ModelTrainingManagerConfig. It must be of optimizer type
    print("\nModelTrainingManagerConfig instance:", trainerConfig)
    print("\nDict of ModelTrainingManagerConfig instance:",
          trainerConfig.getConfigDict())

    # Define model training manager instance
    trainer = ModelTrainingManager(
        model=model, lossFcn=lossFcn, config=trainerConfig)
    print("\nModelTrainingManager instance:", trainer)

    # Define dataloader index for training
    train_loader, validation_loader = DefineDataloaders()

    dataloaderIndex = DataloaderIndex(train_loader, validation_loader)
    # Set dataloaders for training and validation
    trainer.setDataloaders(dataloaderIndex)

    # Perform training and validation of model
    trainer.trainAndValidate()

    # CHECK: versus TrainAndValidateModel
    # model2 = copy.deepcopy(model).to(device)
    # optimizer2 = torch.optim.Adam(model2.parameters(), lr=initial_lr, fused=fused)
    # for epoch in range(numOfEpochs):
    #    print(f"Epoch TEST: {epoch}/{numOfEpochs-1}")
    #    TrainModel(dataloaderIndex.getTrainLoader(), model2, lossFcn, optimizer2, 0)
    #    ValidateModel(dataloaderIndex.getValidationLoader(), model2, lossFcn)

    for param1, param2 in zip(model.parameters(), trainer.model.parameters()):
        if not (torch.equal(param1, param2)) or not (param1 is param2):
            raise ValueError(
                "Model parameters are not the same. optimizer does not update trainer.model!")



if __name__ == '__main__':
    main()
