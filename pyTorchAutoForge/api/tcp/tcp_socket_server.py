# Python imports
import socket
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Union
from enum import Enum
from torch import Tensor
import threading
import time


# TODO
class tcp_socket_server():
    def __init__(self, host: str, port: int, dataProcessorObj: Any):
        self.host = host
        self.port = port

# %% UNIT TESTS
def main():
    pass


if __name__ == "__main__":
    main()