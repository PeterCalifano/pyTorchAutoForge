# Import modules
import torch
import pyTorchAutoForge

from pyTorchAutoForge.optimization.ModelTrainingManager import ModelTrainingManagerConfig, ModelTrainingManager

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

    # Define model training manager config  (dataclass init)
    initial_lr = 1E-4

    trainerConfig = ModelTrainingManagerConfig(initial_lr=initial_lr)
    print(trainerConfig)

    # Define model training manager
    modelTrainingManager = ModelTrainingManager(trainerConfig)



if __name__ == '__main__':
    main()