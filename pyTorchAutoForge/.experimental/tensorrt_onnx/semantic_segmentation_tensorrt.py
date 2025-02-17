# %% Model preparation and testing stage
# Prepare torch model
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import urllib
import torch
model = torch.hub.load('pytorch/vision:v0.10.0',
                       'fcn_resnet50', pretrained=True)
model.eval()

# Download an example image from the pytorch website
url, filename = (
    "https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
try:
    urllib.URLopener().retrieve(url, filename)
except Exception as e:
    print(f"An error occurred: {e}")
    urllib.request.urlretrieve(url, filename)

# Test in pytorch
input_image = Image.open(filename)
input_image = input_image.convert("RGB")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
# Create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

# Move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# Show result
# Create a color palette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# Plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()
                    ).resize(input_image.size)
r.putpalette(colors)

plt.imshow(r)
# plt.show()

# %% Conversion stage
# Export model to ONNx
# TODO


# %% TensorRT engine creation stage
# TODO

# %% Inference stage using TensorRT engine
# TODO