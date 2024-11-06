from torch import nn
import torch
import torch.nn.functional as F

def create_sample_tracedModel_():

    class SampleCNN(nn.Module):
        def __init__(self):
            super(SampleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, 1)
            self.conv2 = nn.Conv2d(8, 16, 3, 1)
            self.fc1 = nn.Linear(16 * 62 * 62, 10)

        # Forward pass function
        def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                return x

    model = SampleCNN()
    
    # Create a random input tensor
    input_tensor = torch.randn(1, 3, 256, 256)

    # Trace the model
    traced_model = torch.jit.trace(model, input_tensor)

    # Save the traced model
    traced_model.save("sample_cnn_traced.pt")

if __name__ == '__main__':
    create_sample_tracedModel_()

