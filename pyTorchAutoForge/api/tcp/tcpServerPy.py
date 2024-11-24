"""! Prototype TCP server script created by PeterC - 15-06-2024"""
# NOTE: the current implementation allows one request at a time, since it is thought for the evaluation of torch models in MATLAB.

# Python imports
import socketserver
import numpy as np
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum

# Check documentation page before coding: https://docs.python.org/3/library/abc.html
class DataProcessingBaseFcn(ABC):
    # TODO: class to constraint implementation of data processing functions DataProcessor uses (sort of abstract class)
    def __init__(self) -> None:
        pass
        
    @abstractmethod
    def process(self, inputData):
        pass


class PreProcessingMode(Enum):
    '''Enum class for data processing modes'''
    NONE = 0
    TENSOR = 1


# %% Data processing function wrapper as generic interface in RequestHandler for TCP servers - PeterC - 15-06-2024
class DataProcessor():
    '''Data processing function wrapper as generic interface in RequestHandler for TCP servers. Input/output for numerical data: numpy.ndarray'''
    def __init__(self, processDataFcn:callable, inputTargetType:Any = np.float32, BufferSizeInBytes:int = -1, ENDIANNESS:str='little', 
                 DYNAMIC_BUFFER_MODE: bool = False, PRE_PROCESSING_MODE: PreProcessingMode = PreProcessingMode.NONE) -> None:
        '''Constructor'''
        self.processDataFcn = processDataFcn
        self.inputTargetType = inputTargetType
        self.BufferSizeInBytes = BufferSizeInBytes
        self.DYNAMIC_BUFFER_MODE = DYNAMIC_BUFFER_MODE
        self.ENDIANNESS = ENDIANNESS
        self.PRE_PROCESSING_MODE = PreProcessingMode.NONE

    def process(self, inputData):
        '''Processing method running specified processing function'''

        # Decode inputData
        decodedData, numBatches = self.decode(inputData)  # DEVNOTE: numBatches will now be directly "encoded" in tensor shape

        if self.PRE_PROCESSING_MODE == PreProcessingMode.TENSOR:
            # Convert input data to tensor shape
            decodedData = self.AdaptiveBufferToTensor(decodedData)
        
        # Execute processing function
        processedData = self.processDataFcn(decodedData, numBatches) # DEVNOTE TODO: replace by standard class method call
        # TODO: replace temporary input with a structured type like a dict to avoid multiple inputs and keep the function interface generic --> better to define a data class?
       
        if self.PRE_PROCESSING_MODE == PreProcessingMode.TENSOR:
            # Convert processed data to adaptive buffer
            processedData = self.TensorToAdaptiveBuffer(processedData)

        return self.encode(processedData)
    
    def decode(self, inputData): # REQUIRES UPDATE
        '''Data conversion function from raw bytes stream to specified target numpy type'''
        if not isinstance(inputData, self.inputTargetType):
            try:
                numBatches = int.from_bytes(inputData[:4], self.ENDIANNESS) # TO REMOVE
                print(f"Received number of batches:\t{numBatches}")
                dataArray = np.array(np.frombuffer(inputData[4:], dtype=self.inputTargetType), dtype=self.inputTargetType)
                print(f"Received data array:\t{dataArray}")
                
            except Exception as errMsg:
                raise TypeError('Data conversion from raw data array to specified target type {targetType} failed with error: \n'.format(targetType=self.inputTargetType) + str(errMsg))
        return dataArray, numBatches
    
    def encode(self, processedData):
        '''Data conversion function from numpy array to raw bytes stream'''
        return processedData.tobytes()
    
    def AdaptiveBufferToTensor(self, inputData):
        '''Function to convert input data to tensor shape'''
        raise NotImplementedError('AdaptiveBufferToTensor method not implemented in DataProcessor class!')
        pass

    def TensorToAdaptiveBuffer(self, processedData):
        '''Function to convert tensor to adaptive buffer'''
        raise NotImplementedError('TensorToAdaptiveBuffer method not implemented in DataProcessor class!')
        pass


# %% Request handler class - PeterC + GPT4o- 15-06-2024
class pytcp_requestHandler(socketserver.BaseRequestHandler):
    '''Request Handler class for tcp server'''
    def __init__(self, request, client_address, server, DataProcessor:DataProcessor, ENDIANNESS:str='little'):
        ''''Constructor'''
        self.DataProcessor     = DataProcessor # Initialize DataProcessing object for handle
        self.BufferSizeInBytes = DataProcessor.BufferSizeInBytes

        assert self.BufferSizeInBytes > 0, "Buffer size must be greater than 0! You probably did not set it or set it to a negative value."

        if hasattr(DataProcessor, 'DYNAMIC_BUFFER_MODE'):
            self.DYNAMIC_BUFFER_MODE = DataProcessor.DYNAMIC_BUFFER_MODE
        else:
            self.DYNAMIC_BUFFER_MODE = False

        if hasattr(DataProcessor, 'ENDIANNESS'):
            self.ENDIANNESS = DataProcessor.ENDIANNESS
        else:
            self.ENDIANNESS = ENDIANNESS

        super().__init__(request, client_address, server)

    def handle(self) -> None:
        '''Function handling request from client'''
        print(f"Handling request from client: {self.client_address}")
        try:
            while True:
                # Read the length of the data (4 bytes) specified by the client
                bufferSizeFromClient = self.request.recv(4)
                if not bufferSizeFromClient:
                    break
                bufferSize = int.from_bytes(bufferSizeFromClient, self.ENDIANNESS) # NOTE: MATLAB writes as LITTLE endian

                # Print received length bytes for debugging a
                print(f"Received length bytes: {bufferSizeFromClient}", ", ", f"Interpreted length: {bufferSize}")               
            
                bufferSizeExpected = self.BufferSizeInBytes

                # Read the entire data buffer
                dataBuffer = b''
                while len(dataBuffer) < bufferSize:
                    packet = self.request.recv(bufferSize - len(dataBuffer))
                    if not packet:
                        break
                    dataBuffer += packet

                # SERVER SHUTDOWN COMMAND HANDLING
                if len(dataBuffer) == 8 and dataBuffer.decode('utf-8'.strip().lower()) == 'shutdown':
                    print("Shutdown command received. Shutting down server...")
                    # Shut down the server
                    self.server.server_close()
                    print('Server is now OFF.')
                    exit()

                # Check if the received data buffer size matches the expected size
                if not(self.DYNAMIC_BUFFER_MODE):
                    print("Expected data buffer size from client:", bufferSizeExpected, "bytes")
                    if not(len(dataBuffer) == bufferSizeExpected):
                        raise BufferError('Data buffer size does not match buffer size by Data Processor! Received message contains {nBytesReceived}'.format(nBytesReceived=len(dataBuffer)) )
                    else:
                        print('Message size matches expected size. Calling data processor...')

                # Data processing handling: move the data to DataProcessor and process according to specified function
                outputDataSerialized = self.DataProcessor.process(dataBuffer)
                # For strings: outputDataSerialized = ("Acknowledge message. Array was received!").encode('utf-8')

                # Get size of serialized output data          
                outputDataSizeInBytes = len(outputDataSerialized)
                print('Sending number of bytes to client:', outputDataSizeInBytes+4)

                # Send the length of the processed data
                self.request.sendall(outputDataSizeInBytes.to_bytes(4, self.ENDIANNESS))

                # Send the serialized output data
                self.request.sendall(outputDataSerialized)
                
        except Exception as e:
            print(f"Error occurred while handling request: {e}")
            
        finally:
            print(f"Connection with {self.client_address} closed")


# %% TCP server class - PeterC - 15-06-2024
class pytcp_server(socketserver.TCPServer):
    allow_reuse_address = True
    '''Python-based custom tcp server class using socketserver module'''

    def __init__(self, serverAddress: tuple[str|bytes|bytearray, int], RequestHandlerClass:pytcp_requestHandler, DataProcessor:DataProcessor, bindAndActivate:bool=True) -> None:
        '''Constructor for custom tcp server'''
        self.DataProcessor = DataProcessor # Initialize DataProcessing object for handle
        super().__init__(serverAddress, RequestHandlerClass, bindAndActivate)
        print('Server opened on (HOST, PORT): (',serverAddress[0],', ',serverAddress[1],')')

    def finish_request(self, request, client_address) -> None:
        '''Function evaluating Request Handler'''
        self.RequestHandlerClass(request, client_address, self, self.DataProcessor)

# %% TEST SCRIPTS

def test_server_implementation():
    pass 

def test_data_processor_base():
    pass

if __name__ == "__main__":
    test_server_implementation()
