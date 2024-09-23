# Import modules
import torch
import pyTorchAutoForge
from pyTorchAutoForge.optimization.ModelTrainingManager import ModelTrainingManagerConfig, ModelTrainingManager, TaskType
from torchvision import models

def main():
    print('\n ------------------ TEST: ModelTrainingManagerConfig and ModelTrainingManager instantiation ------------------')

    # SETTINGS and PARAMETERS
    batch_size = 16*2  # Defines batch size in dataset
    TRAINING_PERC = 0.80
    # outChannelsSizes = [16, 32, 75, 15]
    outChannelsSizes = [16, 32, 75, 15]
    kernelSizes = [3, 1]
    learnRate = 1E-10
    momentumValue = 0.001
    optimizerID = 1  # 0
    device = pyTorchAutoForge.GetDevice()
    exportTracedModel = True


    model = models.resnet18(weights=None).to(device)

    # Define model training manager config  (dataclass init)
    initial_lr = 1E-4
    lossFcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, fused=True)

    trainerConfig = ModelTrainingManagerConfig(tasktype=TaskType.CLASSIFICATION, initial_lr=initial_lr)
    print("\nModelTrainingManagerConfig instance:", trainerConfig)

    print("\nNames of all attributes in ModelTrainingManagerConfig instance:", ModelTrainingManagerConfig.getConfigParamsNames())
    listOfKeys = {key for key in ModelTrainingManagerConfig.getConfigParamsNames() if key in trainerConfig.__dict__.keys()}
    
    print("\nDict of ModelTrainingManagerConfig instance:",
          trainerConfig.getConfigDict())

    # Define model training manager instance
    trainer = ModelTrainingManager(
        model=model, lossFcn=lossFcn, config=trainerConfig)
    print("\nModelTrainingManager instance:", trainer)


if __name__ == '__main__':
    main()