# Script implementing the PyTorch "Training a Classifier" example as exercise by PeterC - 04-05-2024
# Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# Import modules
import torch
from torch import nn
from torch.utils.data import DataLoader # Utils for dataset management, storing pairs of (sample, label)
from torchvision import datasets # Import vision default datasets from torchvision
import torchvision.transforms as transforms # Import transform utilities from torchvision
import datetime
import torchvision
import sys
sys.path.insert(0, "/home/peterc/devDir/MachineLearning_PeterCdev/PyTorch")
import pyTorchAutoForge

from torch.utils.tensorboard import SummaryWriter

# Create tensorboard writer
logDir = './tensorboardTest/'
tensorboardWriter = SummaryWriter(log_dir=logDir, flush_secs=120) # flush_secs determines the refresh rate

# %% PREPARE AND VISUALIZE DATASET 
# Defines a transformation as composition of a sequence of transformations
transformObj = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize( (0.5,0.5,0.5), (0.5,0.5,0.5) )]) 
# NOTE: Normalize defines a normalization of a tensor image with mean and standard deviation for each channel (H,W,C)

batch_size = 5
# Define datasets and pre-process them by applying transformObj transformation
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transformObj) 
validation_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transformObj) 

train_dataset = DataLoader(train_data, batch_size, shuffle=True, num_workers=2) # Set number of workers to 2 (threads)
validation_dataset = DataLoader(validation_data, batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %% OPTTIONAL: PLOT IMAGES
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_dataset)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# %% MODEL ARCHITECTURE DEFINITION
import torch.nn.functional as F # Module to apply activation functions in forward pass instead of defining them in the model class
# NN taking 3-channels images (with colour)

class Conv2dNet(nn.Module):
    def __init__(self, numOfOutputChannels, numOfInputChannels=3, kernelSize=5) -> None:
        super().__init__()
        # NOTE: this model does not have the activation functions specified by the init function as part of the architecture.
        # Instead, the forward pass function defines them. This enables improved code flexibility and reusability.
        self.conv2dL1 = nn.Conv2d(numOfInputChannels, numOfOutputChannels, kernelSize) # Input: Defines a Conv2D layer specifying as a minimum: in and out channels numbers, and the kernel size
        self.maxPool = nn.MaxPool2d(2, 2) # Max pool layer with kernel size and stride as inputs. This is just a re-usable layer.
        self.conv2dL2 = nn.Conv2d(numOfOutputChannels, 16, 5) 
        self.FullyCL3 = nn.Linear(16*5*5, 120) # Takes each output value from Conv2d as input to the 120 neurons of the layer
        self.FullyCL4 = nn.Linear(120, 84)
        self.FullyCL5 = nn.Linear(84, 10) # Output layer

    def forward(self, inputSample):
        value = self.maxPool(F.relu(self.conv2dL1(inputSample))) # Apply layer L1 to get output with ReLU
        value = self.maxPool(F.relu(self.conv2dL2(value))) # Same as L1
        value = torch.flatten(value, 1) # Flatten data to get input to Fully Connected layers
        value = F.relu(self.FullyCL3(value))
        value = F.relu(self.FullyCL4(value))
        logits = self.FullyCL5(value) # Output layer
        return logits

# Instantiate the network model
# NOTE: Number of input channels is 3 --> RGB; number of output channels is a hyperparameter that defines how many kernels (i.e. convolutions) 
# the layer applies to the input channels 

device = pyTorchAutoForge.GetDevice()
ConvNN = Conv2dNet(12, 3, 5).to(device)

# %% TRAINING STEP
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(ConvNN.parameters(), lr=0.001, momentum=0.9)

numOfEpochs = 20
startID = 0

ConvNN.train() # Set the network in training mode --> this has effect only on some modules such as dropouts

for epoch in range(numOfEpochs):
    currentLoss = 0.0

    for ID, data in enumerate(train_dataset, startID):
        # Get (input samples, labels) pairs from the dataset
        inputs, labels = data[0].to(device), data[1].to(device) # Tuple unpacking of data

        # Set gradients to zero
        optimizer.zero_grad()

        # Perform forward step
        predVal = ConvNN(inputs)
        lossFcnVal = criterion(predVal, labels) # Get loss function evaluation
        lossFcnVal.backward() # Propagate gradients
        optimizer.step() # Apply gradients

        # Print statistics
        currentLoss += lossFcnVal.item()
        if ID % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {ID + 1:5d}] loss: {currentLoss / 2000:.3f}')
            currentLoss = 0.0
        tensorboardWriter.add_scalar('Loss function', currentLoss, epoch)
        tensorboardWriter.flush()
        
print("TRAINING LOOP: COMPLETED")
# Save model 
pyTorchAutoForge.SaveModelState(ConvNN, "exampleCNNclassifier")

# %% VALIDATION 
correctOutput = 0
totalDataSize = 0
ConvNN.eval()
with torch.no_grad():
    for data in validation_dataset:
        images, labels = data[0].to(device), data[1].to(device) 
        # Evaluate prediction
        predVal = ConvNN(images)
        # Select the class with the highest energy as prediction
        _, predicted = torch.max(predVal.data, 1)
        totalDataSize += labels.size(0)
        correctOutput += (predicted == labels).sum().item() # Check and accumulate how many correct predictions have been performed

print(f'Accuracy of the network on the test images: {100 * correctOutput // totalDataSize} %')

# %% OPTIONAL: VALIDATION SPLIT PER CLASS
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in train_dataset:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = ConvNN(images)
        _, predictions = torch.max(outputs, 1)
        # Collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# Print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')