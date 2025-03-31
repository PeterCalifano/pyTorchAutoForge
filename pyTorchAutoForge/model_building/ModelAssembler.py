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
from pyTorchAutoForge.utils import GetDevice 
from pyTorchAutoForge.model_building import torchModel
from typing import Union
from torch import nn

# DEVNOTE: verify is tracing now works with this class
class MultiHeadAdapter(nn.Module):
    def __init__(self, numOfHeads: int, headModels:Union[nn.Module, torchModel, nn.ModuleList, nn.ModuleDict]) -> "MultiHeadAdapter":
        self.numOfHeads = numOfHeads
        self.headList = nn.ModuleDict()
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

    def to(self, device):
        for head in self.headList:
            head.to(device)

# DEVNOTE: python features to use:
#class MyClass:
#    def __init__(self, *args: Union[int, float, str]) -> None:
#        for i, arg in enumerate(args):
#            setattr(self, f'attribute_{i}', arg)
#    
#    def display_attributes(self):
#        # Just an example to show the attributes
#        for attr, value in self.__dict__.items():
#            print(f"{attr}: {value}")

# DEVNOTE: try to make it a subclass of nn.Module
class ModelAssembler(nn.Module):
    # DEVNOTE: behaviour for each type is TBC
    def __init__(self, device:str = GetDevice(),  *args: Union[torchModel, nn.Module, nn.ModuleList, nn.ModuleList]) -> "ModelAssembler":
        super(self).__init__()

        for idModule, module in enumerate(args):
            
            # Process each moduel according to instance type
            if isinstance(module, nn.Module):
                module_name = module.__class__.__name__
                setattr(self, f'{module_name}_{idModule}', module)

            elif isinstance(module, nn.ModuleList):
                module_name = f"moduleList_{idModule}" # Note: no way of getting the name of the module list

            elif isinstance(module, nn.ModuleDict):

                # Unpack the module dict and assign to the model
                for key, module in module.items():
                    module_name = f"{key}_{idModule}" 
                    setattr(self, module_name, module)
            else:
                raise TypeError(f"Module type {type(module)} is not supported")
        return self

    def forward(self, X):
        pass # TODO --> Optional

    def getModelAsModule(self):
        
        # Pack all modules into a nn.ModuleDict
        model = nn.ModuleDict()

        for module_name, module in self.__dict__.items():
            model[module_name] = module

        return model


        

        

