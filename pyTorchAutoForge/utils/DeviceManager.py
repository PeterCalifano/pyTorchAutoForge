import torch

# GetDevice:
def GetDevice():
    '''Function to get working device. Used by most modules of pyTorchAutoForge'''
    # TODO: improve method by adding selection of GPU for multi-GPU systems
    device = ("cuda:0"
              if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu")
    # print(f"Using {device} device")
    return device

# Temporary placeholder class (extension wil be needed for future implementations, e.g. multi GPUs)
class DeviceManager():
    def __init__(self):
        pass 
    
    @staticmethod
    def GetDevice(self):
        return GetDevice()