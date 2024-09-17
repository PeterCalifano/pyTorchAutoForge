''' Script created by PeterC to test mlflowand optuna integration library for model monitoring and tracking - 02-07-2024 '''

import optuna
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Import modules
import sys, os, subprocess, time, logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Append paths of custom modules
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/customTorchTools'))
sys.path.append(os.path.join('/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch/LimbBasedNavigationAtMoon'))

import PyTorch.pyTorchAutoForge.pyTorchAutoForge.pyTorchAutoForge as pyTorchAutoForge # Custom torch tools
import numpy as np

def StartMLflowUI(port:int=5000):

    # Start MLflow UI
    os.system('mlflow ui --port ' + str(port))
    process = subprocess.Popen(['mlflow', 'ui', '--port ' + f'{port}', '&'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f'MLflow UI started with PID: {process.pid}, on port: {port}')
    time.sleep(1) # Ensure the server has started
    if process.poll() is None:
        print('MLflow UI is running OK.')
    else:
        raise RuntimeError('MLflow UI failed to start. Run stopped.')

    return process

# %% Datasets loading (global)
# Load CIFAR-10 dataset from torchvision
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# TEST EXAMPLE BY GPT
# %% Optuna model for trial optimization
class SimpleCNN(nn.Module):
    def __init__(self, trial):
        super(SimpleCNN, self).__init__()
        self.layers = nn.Sequential()

        # Convolutional layers
        num_conv_layers = trial.suggest_int('num_conv_layers', 1, 2)

        # NOTE: What are the second inputs to suggest_int? Are they the lower and upper bounds?

        in_channels = 3 # Number of channels in the input image (RGB)

        # Add convolutional blocks to the model up to "num_conv_layers" --> optimization parameter
        for i in range(num_conv_layers):
            
            # Number of output channels for the ith convolutional block
            out_channels = trial.suggest_int(f'filters_{i}', 16, 64)
            # Kernel size for the ith convolutional block
            kernel_size = trial.suggest_int(f'kernel_size_{i}', 3, 5)

            # Add convolutional block to the model. layers.add_module() takes input:(name, module)
            # NOTE: this is easily repurposed for the Model AutoBuilder
            self.layers.add_module(f'conv{i}', nn.Conv2d(in_channels, out_channels, kernel_size))
            self.layers.add_module(f'relu{i}', nn.ReLU())
            self.layers.add_module(f'maxpool{i}', nn.MaxPool2d(2))

            in_channels = out_channels # Number of input channels to conv2d is the number of output channels from the previous conv2d

        self.layers.add_module('flatten', nn.Flatten()) # Add flatten before NN

        # Dense layers
        num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
        in_features = self._get_conv_output((3, 32, 32))

        for i in range(num_dense_layers):
            out_features = trial.suggest_int(f'units_{i}', 32, 256)
            self.layers.add_module(f'fc{i}', nn.Linear(in_features, out_features))
            self.layers.add_module(f'relu_fc{i}', nn.ReLU())
            dropout_rate = trial.suggest_float(f'dropout_rate_{i}', 0.1, 0.5)
            self.layers.add_module(f'dropout{i}', nn.Dropout(dropout_rate))
            in_features = out_features

        self.layers.add_module('output', nn.Linear(in_features, 10))

    def forward(self, x):
        return self.layers(x)

    # Function to compute the shape of the output of the a convolutional layer
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input_data = torch.rand(1, *shape)
            output = self.layers(input_data)
            return output.size(1)

# %% Optuna objective function
def objective(trial):
    # Create new run_IN within the current session
    with mlflow.start_run():

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create the trial model
        model = SimpleCNN(trial).to(device)

        # Select the optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
        lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True) # NOTE: What are the inputs to suggest_float?
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        # NOTE: What is "getattr" function? It should be a pytorch function: likely a method of "model", TBC
    
        # Loss function definition
        criterion = nn.CrossEntropyLoss()

        # MLflow: log trial data
        mlflow.log_param('optimizer', optimizer_name)
        mlflow.log_param('learning_rate', lr)
        mlflow.log_param('num_conv_layers',
                         trial.params.get('num_conv_layers'))
        # NOTE: integration with optuna --> parameters are got directly from the ith trial object under evaluation
        mlflow.log_param('num_dense_layers',
                         trial.params.get('num_dense_layers'))

        for i in range(trial.params.get('num_conv_layers')):
            mlflow.log_param(f'filters_{i}', trial.params.get(f'filters_{i}'))
            mlflow.log_param(
                f'kernel_size_{i}', trial.params.get(f'kernel_size_{i}'))

        for i in range(trial.params.get('num_dense_layers')):
            mlflow.log_param(f'units_{i}', trial.params.get(f'units_{i}'))
            mlflow.log_param(
                f'dropout_rate_{i}', trial.params.get(f'dropout_rate_{i}'))
            
        # Training the current trial model
        for epoch in range(10):
            model.train()
            for batch in train_loader:
                inputs, targets = batch # Unpack tuples
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                # Evaluate model
                outputs = model(inputs)
                # Compute loss
                loss = criterion(outputs, targets)
                # Perform backpropagation
                loss.backward()
                optimizer.step()

            # Model validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            accuracy = correct / total
            
            # Log the accuracy of the model using mlflow
            mlflow.log_metric('Accuracy value', accuracy, step=epoch)

            trial.report(accuracy, epoch) # Report the accuracy of the model to optuna pruner

            if trial.should_prune():
                # Mark the run as killed in mlflow
                mlflow.end_run(status='KILLED')
                raise optuna.TrialPruned() # Raise an optuna exception to stop the trial due to pruning
            
        # MLflow: log trial data (it will have a unique ID)
        # NOTE: this could be matched with the AutoBuilder configuration such that the parameters 
        # of config file are automatically logged by mlflow with their names
        mlflow.end_run(status='FINISHED')

        return accuracy


def main():
    print('---------------------------- TEST SCRIPT: MLflow and Optuna functionalities ----------------------------\n')
    # %% MLflow tracking initialization
    port = 7000
    #StartMLflowUI(port) # Start MLflow UI

    # %% Optuna study configuration
    if not (os.path.exists('optuna_db')):
        os.makedir('optuna_db')

    studyName = 'CIFAR10_CNN_OptimizationExample'
    optunaStudyObj = optuna.create_study(study_name=studyName,
                                         storage='sqlite:///{studyName}.db'.format(studyName=os.path.join('optuna_db', studyName)),
                                         load_if_exists=True,
                                         direction='maximize',
                                         sampler=optuna.samplers.TPESampler(),
                                         pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2,
                                                                                       min_early_stopping_rate=1))
    
    # Mlflow experiment name
    mlflow.set_experiment(studyName)

    # %% Optuna optimization
    optunaStudyObj.optimize(objective, n_trials=100, timeout=1800)
    
    # Print the best trial
    print('Number of finished trials:', len(optunaStudyObj.trials)) # Get number of finished trials
    print('Best trial:')

    trial = optunaStudyObj.best_trial # Get the best trial from study object
    print('  Value: {:.4f}'.format(trial.value)) # Loss function value for the best trial

    # Print parameters of the best trial
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
        

# %% Loss visualization Functions - PeterC - 06-07-2024

# Example usage
# Assuming you have a dataloader `data_loader` and loss function `loss_fn`, call the function:
# plot_loss_landscape(mofrl, data_loader, loss_fn)

def GetRandomTensorDirection(model):
    direction = []
    # For each entry in parameters (a tensor!) in the model, create a random tensor with the same shape
    for param in model.parameters(): # What does parameters() return specifically?
        if param.requires_grad:
            direction.append(torch.randn_like(param))
    return direction

def FilterNormalizeDirections(d1, d2, model):
    norm_d1, norm_d2 = [], []
    for p1, p2, param in zip(d1, d2, model.parameters()):
        if param.requires_grad:
            norm = torch.norm(param)
            norm_d1.append(p1 / torch.norm(p1) * norm)
            norm_d2.append(p2 / torch.norm(p2) * norm)
    return norm_d1, norm_d2

def DisplaceModelParamsAlongDirections(model, direction, alpha):
    index = 0
    for param in model.parameters():
        if param.requires_grad:
            param.data.add_(direction[index], alpha=alpha)
            index += 1

# Auxiliary function (do not move to customTorchTools)
def calculate_loss(model, data_loader, loss_fn):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss += loss_fn(outputs, targets).item()
    return loss / len(data_loader)

def Plot2DlossLandscape(model, dataloader, lossFcn, dir1Range=(-1, 1), dir2Range=(-1, 1), gridSize=0.1):
    # Get two sets of random directions and normalize them using filter normalization
    d1 = GetRandomTensorDirection(model)
    d2 = GetRandomTensorDirection(model)

    d1, d2 = FilterNormalizeDirections(d1, d2, model)

    # Pre-allocate plot grid
    gammaVals = np.arange(dir1Range[0], dir1Range[1], gridSize)
    etaVals = np.arange(dir2Range[0], dir2Range[1], gridSize)
    losses = np.zeros((len(gammaVals), len(etaVals)))

    originalModelState = [param.clone() for param in model.parameters()]

    for i, x in enumerate(gammaVals):
        for j, y in enumerate(etaVals):

            # Compute displaced model parameters along the two directions for a given x and y
            DisplaceModelParamsAlongDirections(model, d1, x)
            DisplaceModelParamsAlongDirections(model, d2, y)

            # Evaluate loss using displaced model
            losses[i, j] = calculate_loss(model, dataloader, lossFcn)

            # NOTE: Not sure what this does?
            for param, original_param in zip(model.parameters(), originalModelState):
                param.data.copy_(original_param)

    # Generate mesh grid for plotting
    X, Y = np.meshgrid(gammaVals, etaVals)
    Z = losses.T  # Transpose to match the correct orientation

    # Plot 2D contour of the loss landscape
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.title('Loss Landscape')
    plt.show()


# %% Main call
if __name__ == "__main__":
    main()