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

from functools import partial

def objective(trial: optuna.Trial, train_dataset, validation_dataset, numOfEpochs):

    # Get model backbone from torchvision
    # All models in torchvision.models for classification are trained on ImageNet, thus have 1000 classes as output
    model = DefineModel()
    device = GetDevice()
    model.to(device)

    lossFcn, initial_lr = DefineOptimStrategy(trial)

    fused = True if device == "cuda:0" else False
    optimizer = torch.optim.Adam( model.parameters(), lr=initial_lr, fused=fused)

    # Define dataloaders
    batch_size = trial.suggest_int('batch_size', 32, 512)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    validation_loader = DataLoader(validation_dataset,
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Define dataloader index for training
    dataloaderIndex = DataloaderIndex( train_loader, validLoader=validation_loader)  # Use split

    with mlflow.start_run():

        # MLflow: log trial data
        mlflow.log_param('learning_rate', initial_lr)
        mlflow.log_param(f'Dense layer size', model.fc.in_features)
                
        # Define model training manager config  (dataclass init)
        trainerConfig = ModelTrainingManagerConfig(tasktype=TaskType.CLASSIFICATION,
                                               initial_lr=initial_lr, lr_scheduler=None,
                                               num_of_epochs=numOfEpochs, optimizer=optimizer,
                                                batch_size=batch_size)

        # Define model training manager instance
        trainer = ModelTrainingManager(
            model=model, lossFcn=lossFcn, config=trainerConfig)
        print("\nModelTrainingManager instance:", trainer)

        # Set dataloaders for training and validation
        trainer.setDataloaders(dataloaderIndex)

        # Perform training and validation of model
        trainer.trainAndValidate()

        # Get score for optuna
        if trainer.bestModel is not None:
            stats = trainer.evalBestAccuracy()
            average_loss = stats['average_loss'] = average_loss
        else:
            average_loss = 1E10

    return average_loss

def main():

    # Set mlflow experiment
    studyName = 'ImageClassificationCIFAR10_HparamOptim_Example'
    mlflow.set_experiment(studyName)

    numOfEpochs = 25
    NUM_TRIALS = 10

    # Define dataloader index for training
    train_loader, validation_loader = DefineDataloaders()

    # DEFINE STUDY
    # ACHTUNG: number of startup trials if relevant in determining the performance of the sampler.
    # This is because it determines the initial distributions from which the sampler starts working, i.e. determines which clusters of hyperparameters are more likely sampled.

    sampler = optuna.samplers.TPESampler(n_startup_trials=2, seed=10) # What does the multivariate option do?
    # sampler = optuna.samplers.GPSampler(n_startup_trials=25)

    optunaStudyObj = optuna.create_study(study_name=studyName,
                                         load_if_exists=True,
                                         direction='minimize',
                                         sampler=sampler,
                                         pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', reduction_factor=10,
                                                                                       min_early_stopping_rate=5))

    # RUN STUDY
    NUM_OF_JOBS = 1
    objective_func = partial(objective, train_dataset=train_loader.dataset, validation_dataset=validation_loader.dataset, numOfEpochs=numOfEpochs)

    optunaStudyObj.optimize( objective_func, n_trials=NUM_TRIALS,  timeout=2*3600, n_jobs=NUM_OF_JOBS)

    # Print the best trial

    # Get number of finished trials
    print('Number of finished trials:', len(optunaStudyObj.trials))
    print('Best trial:')

    trial = optunaStudyObj.best_trial  # Get the best trial from study object
    # Loss function value for the best trial
    print('  Value: {:.4f}'.format(trial.value))

    # Print parameters of the best trial
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

if __name__ == '__main__':
    main()
