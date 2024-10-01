"""
# The BASIC IDEA:
# Get model backbone from torchvision
# All models in torchvision.models for classification are trained on ImageNet, thus have 1000 classes as output
model = models.resnet18(weights=None)
print(model)
device = GetDevice()
# Therefore, let's modify the last layer! (Named fc)
numInputFeatures = model.fc.in_features  # Same as what the model uses
numOutClasses = 100  # Selected by user
model.fc = (nn.Linear(in_features=numInputFeatures,
                      out_features=numOutClasses, bias=True))  # 100 classes in our case
model.to(device)
print(model)  # Check last layer now
# Test assembly of models
model2 = (nn.Linear(in_features=numOutClasses,
                    out_features=10, bias=True))
model_assembly = nn.Sequential(*[model, model2])
print(model_assembly)
exit(0)
"""

from  pyTorchAutoForge import torchModel
from typing import Union
from torch import nn


class MultiHeadAdapter():
    def __init__(self, numOfHeads: int, headModels:Union[nn.Module, torchModel, nn.ModuleList, nn.ModuleDict]) -> "MultiHeadAdapter":
        self.numOfHeads = numOfHeads
        self.headList = []
        self.headNames = None

        if isinstance(headModels, nn.ModuleList):
            self.headList = headModels
            for i in range(numOfHeads):
                self.headNames.append(f"head_{i}")
        elif isinstance(headModels, nn.ModuleDict):
            self.headList = headModels.values()
            self.headNames = headModels.keys()
        elif isinstance(headModels, (nn.Module, torchModel)):
            self.headList = [headModels]
            self.headNames = ["head_0"]
        
    def forward(self, Xfeatures):
        return [head(Xfeatures) for head in self.headList]
        

class ModelAssembler():
    def __init__(self) -> None:
        super().__init__()
        # TODO

        