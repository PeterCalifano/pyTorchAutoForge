# Import modules
import torch
import pytest
from pyTorchAutoForge.optimization.ModelTrainingManager import ModelTrainingManagerConfig, ModelTrainingManager, TaskType
from pyTorchAutoForge.datasets import DataloaderIndex
import tempfile

from torchvision import models
import optuna, mlflow

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for this test.")
def test_ModelTrainingManager():

    from torchvision import transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader

    def _DefineModel(trial: optuna.Trial | None = None):

        model = models.resnet18(weights=False)

        # Therefore, let's modify the last layer! (Named fc)
        if trial is not None:
            num_input_features = trial.suggest_int(
                'num_input_features', 2, 1000)
        else:
            num_input_features = model.fc.in_features  # Same as what the model uses

        num_out_classes = 10  # Selected by user
        model.fc = (torch.nn.Linear(in_features=num_input_features,
                              out_features=num_out_classes, bias=True))  # 100 classes in our case

        return model

    def _DefineDataloaders(trial: optuna.Trial | None = None):
        # Build small synthetic datasets for a fast smoke test
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.FakeData(
            size=128,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )
        validation_dataset = datasets.FakeData(
            size=64,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )

        # Define dataloaders
        if trial is not None:
            batch_size = trial.suggest_int('batch_size', 32, 512)
        else:
            batch_size = 32

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
            
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_loader, validation_loader

    def _DefineOptimStrategy(trial: optuna.Trial | None = None):
        if trial is not None:
            initial_lr = trial.suggest_loguniform('initial_lr', 1E-2, 1E-1)

        else:
            initial_lr = 1E-3

        lossFcn = torch.nn.CrossEntropyLoss()

        return lossFcn, initial_lr

    previous_tracking_uri = mlflow.get_tracking_uri()
    mlflow_tmp_dir = tempfile.TemporaryDirectory()
    try:
        mlflow.set_tracking_uri(f"file://{mlflow_tmp_dir.name}")
        # Set mlflow experiment
        mlflow.set_experiment('test_ModelTrainingManager')

        # Get model backbone from torchvision
        # All models in torchvision.models for classification are trained on ImageNet, thus have 1000 classes as output
        model = _DefineModel()
        device = 'cpu'
        model.to(device)
        print(model)  # Check last layer

        # Define loss function and optimizer
        lossFcn, initial_lr = _DefineOptimStrategy()
        num_epochs = 1

        fused = True if device == "cuda:0" else False
        # optimizer = torch.optim.Adam(
        #    model.parameters(), lr=initial_lr, fused=fused)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=initial_lr, momentum=0.9)

        # optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)

        # Define dataloader index for training
        train_loader, validation_loader = _DefineDataloaders()  # With default batch size
        dataloaderIndex = DataloaderIndex(train_loader, validation_loader)

    # CHECK: versus TrainAndValidateModel
    # model2 = copy.deepcopy(model).to(device)
    # optimizer2 = torch.optim.Adam(model2.parameters(), lr=initial_lr, fused=fused)
    # for epoch in range(numOfEpochs):
    #    print(f"Epoch TEST: {epoch}/{numOfEpochs-1}")
    #    TrainModel(dataloaderIndex.getTrainLoader(), model2, lossFcn, optimizer2, 0)
    #    ValidateModel(dataloaderIndex.getValidationLoader(), model2, lossFcn)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)

        if train_loader is None or validation_loader is None:
            raise ValueError("Dataloader index is None. Check dataloaders.")
    
        # Define model training manager config  (dataclass init)
        trainerConfig = ModelTrainingManagerConfig(tasktype=TaskType.CLASSIFICATION,
                                                   initial_lr=initial_lr,
                                                   lr_scheduler=lr_scheduler,
                                                   num_of_epochs=num_epochs,
                                                   optimizer=optimizer,
                                                   batch_size=train_loader.batch_size,
                                                   mlflow_logging=False)

        print("\nModelTrainingManagerConfig instance:\n", trainerConfig)
        print("\nDict of ModelTrainingManagerConfig instance:\n",
              trainerConfig.get_config_dict())

        # Test overriding of optimizer
        optimizer = torch.optim.SGD(
            model.parameters(), lr=initial_lr, momentum=0.9)

        # Define model training manager instance
        trainer = ModelTrainingManager(
            model=model, lossFcn=lossFcn, config=trainerConfig)
        print("\nModelTrainingManager instance:\n\n", trainer)

        # Set dataloaders for training and validation
        trainer.setDataloaders(dataloaderIndex)

        # Perform training and validation of model
        trainer.trainAndValidate()
    finally:
        mlflow_tmp_dir.cleanup()
        mlflow.set_tracking_uri(previous_tracking_uri)

if __name__ == '__main__':
    test_ModelTrainingManager()
