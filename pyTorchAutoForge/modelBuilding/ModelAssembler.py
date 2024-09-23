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