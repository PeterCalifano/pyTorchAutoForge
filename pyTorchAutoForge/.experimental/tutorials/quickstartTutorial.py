# Script implementing the PyTorch quick-start example as exercise by PeterC - 04-05-2024
# Reference: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

# Import modules
import torch
from torch import nn
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
from torchvision.transforms import ToTensor # Utils
import datetime
import numpy as np

# %% DATASET MANAGEMENT
# Full list of datasets in torchvision: https://pytorch.org/vision/stable/datasets.html
# All Dataset objects have two arguments: transform and target_transform to apply changes to samples and labels, respectively.

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
) 
# Dataset object, needs to be passed to DataLoader to get values. This wraps an iterable over our dataset, and supports automatic 
# batching, sampling, shuffling and multiprocess data loading

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
) 


batch_size = 64 # Defines batch size in dataset
training_dataset = DataLoader(training_data, batch_size, shuffle=False)
test_dataset = DataLoader(test_data, batch_size, shuffle=False) # NOTE: the dataloader is iterable, meaning that for loops can be built upon it

for Xi,Yi in training_dataset:
    print(f"Shape of Xi [N, C, H, W]: {Xi.shape}")
    print(f"Shape of Yi: {Yi.shape} {Yi.dtype}")
    break


# %% DEVICE SELECTION
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %% MODEL CREATION AS CLASS
# Define class inheriting from nn.Module parent class
mnistPix = 28
inputSize = mnistPix # Input size for input layer
layersSize = [1024, 1024, 10] # List of output sizes for each layer

# TO VERIFY: can I do a NN auto constructor in PyTorch? --> definition of layer types + input size + output sizes for each layer?

class NeuralNet(nn.Module):
    # CONSTRUCTOR
    def __init__(self, layersSize, netName_in='unnamed') -> None:
        super().__init__() # Calling the class constructor of the parent class nn.Module
        self.flatten = nn.Flatten() # Define flatten function for NeuralNet class
        netName = netName_in # Assign internal name to the net (just for fun)

        self.architecture = nn.Sequential(
            nn.Linear(inputSize*inputSize, layersSize[0]),
            nn.ReLU(),
            nn.Linear(layersSize[0], layersSize[1]),
            nn.ReLU(),
            nn.Linear(layersSize[1], layersSize[2])
        ) # Define architecture of the network

    # FORWARD PASS METHOD
    def forward(self, xInput): 
        xInput = self.flatten(xInput) # Flatten input vector to 1D 
        logits = self.architecture(xInput) # Evaluate model at input
        return logits

model = NeuralNet(layersSize).to(device) # Create instance of model using device 
print(model)

# %% MODEL TRAINING
# NOTE on PyTorch modules:
# nn module contains building blocks for neural nets architectures
# optim module contains optimizers

# Define loss function
# torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)
loss_fn = nn.CrossEntropyLoss() 

learnRate = 1E-3
# Define optimizer object specifying model instance parameters and optimizer parameters
optimizer = torch.optim.SGD(model.parameters(), lr=learnRate, momentum=0) 

# Training function over entire dataset (single epoch)
def trainModel(dataloader:DataLoader, model:nn.Module, lossFcn, optimizer):
    size=len(dataloader.dataset) # Get size of dataloader dataset object
    model.train() # Set model instance in training mode ("informing" backend that the training is going to start)
    totalLoss = 0.0

    for batch, (X, Y) in enumerate(dataloader): # Recall that enumerate gives directly both ID and value in iterable object

        X, Y = X.to(device), Y.to(device) # Define input, label pairs for target device
        
        # Perform FORWARD PASS
        predVal = model(X) # Evaluate model at input
        loss = lossFcn(predVal, Y) # Evaluate loss function to get loss value (this returns loss function instance, not a value)

        # Perform BACKWARD PASS
        loss.backward() # Compute gradients
        optimizer.step() # Apply gradients from the loss
        optimizer.zero_grad() # Reset gradients for next iteration

        if batch % 100 == 0: # Print loss value every 100 steps
            lossvalue, currentStep = loss.item(), (batch + 1) * len(X)
            print(f"loss: {lossvalue:>7f}  [{currentStep:>5d}/{size:>5d}]")
    
        totalLoss += loss.item()
    return totalLoss 
# %% MODEL VALIDATION

def validateModel(dataloader:DataLoader, model:nn.Module, lossFcn):
    size = len(dataloader.dataset) 
    numberOfBatches = len(dataloader)
    model.eval() # Set the model in evaluation mode
    testLoss, correctOuputs = 0, 0 # Accumulation variables

    with torch.no_grad(): # Tell torch that gradients are not required
        for X,Y in dataloader:

            X, Y = X.to(device), Y.to(device) # Define input, label pairs for target device
            # Perform FORWARD PASS
            predVal = model(X) # Evaluate model at input
            testLoss += lossFcn(predVal, Y).item() # Evaluate loss function and accumulate
            # Determine if prediction is correct and accumulate
            # Explanation: get largest output logit (the predicted class) and compare to Y. 
            # Then convert to float and sum over the batch axis, which is not necessary if size is single prediction
            correctOuputs += (predVal.argmax(1) == Y).type(torch.float).sum().item() 

    testLoss/=numberOfBatches # Compute batch size normalized loss value
    correctOuputs /= size # Compute percentage of correct classifications over batch size
    print(f"Test Error: \n Accuracy: {(100*correctOuputs):>0.1f}%, Avg loss: {testLoss:>8f} \n")
    return testLoss, correctOuputs

# %% PERFORM TRAINING EPOCHS
epochNum = 500
trainLoss_list = np.zeros([epochNum, 1])
testLoss_list = np.zeros([epochNum, 1])
correctOuputs_list = np.zeros([epochNum, 1])

for epochID in range(epochNum):
    print(f"Epoch {epochID}\n-------------------------------")
    # Do training over all batches
    trainLoss_list[epochID] = trainModel(training_dataset, model, loss_fn, optimizer) 
    # Do validation over all batches
    testLoss_list[epochID], correctOuputs_list[epochID] = validateModel(test_dataset, model, loss_fn) 

# %% OPTIONAL: MODEL STATE SAVING
def saveModelState(model:nn.Module):
    import os.path
    if not(os.path.isdir('./testModels')):
        os.mkdir('testModels')

    currentTime = datetime.datetime.now()
    formattedTimestamp = currentTime.strftime('%d-%m-%Y_%H-%M') # Format time stamp as day, month, year, hour and minute

    filename = "testModels/trainedModel_" + formattedTimestamp
    print("Saving PyTorch Model State to", filename)
    torch.save(model.state_dict(), filename) # Save model as internal torch representation

saveModelState(model)

# %% OPTIONAL: MODEL LOADING
# model = NeuralNet().to(device) # Define empty model instance
# model.load_state_dict(torch.load("model.pth")) # Load saved state (parameters) into model instance