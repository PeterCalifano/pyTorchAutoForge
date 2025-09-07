import pytest
import test
from pyTorchAutoForge.utils.DeviceManager import GetDeviceMulti, GetCudaAvailability

#def test_GetDevice_():
#    # TODO
#    # Test the GetDevice function
#    assert GetDeviceMulti() == "cuda:0" or GetDeviceMulti(
#    ) == "cpu" or GetDeviceMulti() == "mps"
#
#    print("GetDevice() test passed. Selected device: ", GetDeviceMulti())

def test_GetCudaAvailability(): 

    GetCudaAvailability()

if __name__ == "__main__":
    import sys
    import os

    #test_GetDevice_()
    test_GetCudaAvailability()
