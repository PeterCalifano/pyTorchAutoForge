"""! Prototype TCP server script created by PeterC - 15-06-2024"""
# NOTE: the current implementation allows one request at a time, since it is thought for the evaluation of torch models in MATLAB.

# Python imports
import socketserver
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Union
from enum import Enum

import socket
import threading

# Check documentation page before coding: https://docs.python.org/3/library/abc.html
class DataProcessingBaseFcn(ABC):
    # TODO: class to constraint implementation of data processing functions DataProcessor uses (sort of abstract class)
    def __init__(self) -> None:
        pass
        
    @abstractmethod
    def process(self, inputData):
        pass


class ProcessingMode(Enum):
    '''Enum class for data processing modes'''
    NONE = 0
    TENSOR = 1
    MULTI_TENSOR = 2


# %% Data processing function wrapper as generic interface in RequestHandler for TCP servers - PeterC - 15-06-2024
class DataProcessor():
    '''Data processing function wrapper as generic interface in RequestHandler for TCP servers. Input/output for numerical data: numpy.ndarray'''
    def __init__(self, processDataFcn:callable, inputTargetType:Any = np.float32, BufferSizeInBytes:int = -1, ENDIANNESS:str='little', 
                 DYNAMIC_BUFFER_MODE: bool = False, PRE_PROCESSING_MODE: ProcessingMode = ProcessingMode.MULTI_TENSOR) -> None:
        '''Constructor'''
        self.processDataFcn = processDataFcn
        self.inputTargetType = inputTargetType
        self.BufferSizeInBytes = BufferSizeInBytes
        self.DYNAMIC_BUFFER_MODE = DYNAMIC_BUFFER_MODE
        self.ENDIANNESS = ENDIANNESS
        self.PROCESSING_MODE = ProcessingMode.MULTI_TENSOR

    def process(self, inputDataBuffer: bytes) -> bytes:
        '''Processing method running specified processing function'''

        # Decode inputData
        decodedData, numBatches = self.decode(inputDataBuffer)  # DEVNOTE: numBatches will now be directly "encoded" in tensor shape

        # Execute processing function
        processedData = self.processDataFcn(decodedData, numBatches) # DEVNOTE TODO: replace by standard class method call
        # TODO: replace temporary input with a structured type like a dict to avoid multiple inputs and keep the function interface generic --> better to define a data class?
       
        return self.encode(processedData)
    
    def decode(self, inputDataBuffer):
        '''Data conversion function from raw bytes stream to specified target numpy type with specified shape'''
        if not isinstance(inputDataBuffer, self.inputTargetType):

            if self.PROCESSING_MODE == ProcessingMode.TENSOR:
                # Convert input data to tensor shape
                [dataArray, dataArrayShape] = self.BytesBufferToTensor(inputDataBuffer)
                print(f"Received tensor of shape:\t{dataArrayShape}")

            elif self.PROCESSING_MODE == ProcessingMode.MULTI_TENSOR:

                # Convert input data to multi-tensor list
                [dataArray, dataArrayShape, numOfTensors] = self.BytesBufferToMultiTensor(inputDataBuffer)
                print(f"Received list of tensors of length:\t{numOfTensors}")

            else:
                #dataArray = np.array(np.frombuffer(inputDataBuffer[4:], dtype=self.inputTargetType), dtype=self.inputTargetType)
                dataArrayBuffer = inputDataBuffer
                try:
                    dataArray = np.array(np.frombuffer(dataArrayBuffer, dtype=self.inputTargetType), dtype=self.inputTargetType)
                    dataArrayShape = dataArray.shape

                except Exception as errMsg:
                    raise TypeError('Data conversion from raw data array to specified target type {targetType} failed with error: {errMsg}\n'.format(
                        targetType=self.inputTargetType, errMsg = str(errMsg)))
            

        return dataArray, dataArrayShape
    
    def encode(self, processedData):
        '''Data conversion function from numpy array to raw bytes stream'''
        if self.PROCESSING_MODE == ProcessingMode.TENSOR:
            # Convert processed data to tensor-convention buffer
            processedData = self.TensorToBytesBuffer(processedData)
            return processedData
        
        elif self.PROCESSING_MODE == ProcessingMode.MULTI_TENSOR:

            # Convert processed data multi-tensor to tensor-convention buffer
            processedData = self.MultiTensorToBytesBuffer(processedData)
            return processedData
        
        else:
            return processedData.tobytes()
    
    def BytesBufferToTensor(self, inputDataBuffer: bytes) -> tuple[np.ndarray, tuple[int]]:
        """Function to convert input data message from bytes to tensor shape. The buffer is expected to be in the following format:
        - 4 bytes: number of dimensions (int)
        - 4 bytes per dimension: shape of tensor (int)
        - remaining bytes: flattened tensor data (float32), column-major order

        Args:
            inputDataBuffer (bytes): Input bytes buffer

        Returns:
            tuple[np.ndarray, tuple[int]]: Tuple containing the tensor data and its shape
        """

        # Get number of dimensions
        numOfDims = int.from_bytes(inputDataBuffer[:4], self.ENDIANNESS)  #
        # Get shape of tensor ( TO VERIFY IF THIS WORKS) 
        dataArrayShape = tuple(int.from_bytes(inputDataBuffer[4:8+4*(idx)], self.ENDIANNESS) for idx in range(numOfDims))
        # Convert buffer to numpy array with specified shape (REQUIRES TESTING)
        dataArray = np.array(np.frombuffer(inputDataBuffer[4+4*numOfDims:], dtype=self.inputTargetType), dtype=self.inputTargetType).reshape(dataArrayShape)
        
        return dataArray, dataArrayShape



    def TensorToBytesBuffer(self, processedData: np.ndarray) -> bytes:
        """Function to convert input tensor to buffer message. The buffer is generated according to the following format:
        - 4 bytes: message length (int)
        - 4 bytes: number of dimensions (int)
        - 4 bytes per dimension: shape of tensor (int)
        - remaining bytes: flattened tensor data (float32), column-major order reshaping

        Args:
            processedData (np.ndarray): Input tensor data

        Raises:
            TypeError: If input data is not a numpy array

        Returns:
            bytes: Output bytes buffer
        """

        if not isinstance(processedData, np.ndarray):
            raise TypeError('Input data must be a numpy array.')
                            
        # Get shape of tensor
        dataArrayShape = processedData.shape
        # Get number of dimensions
        numOfDims = len(dataArrayShape)
        # Convert column-major flattened numpy array to buffer (REQUIRES TESTING)
        dataArrayBuffer = processedData.reshape(-1, order='F').tobytes() 

        # Create buffer with shape and data
        outputBuffer = numOfDims.to_bytes(4, self.ENDIANNESS) + (b''.join([dim.to_bytes(4, self.ENDIANNESS) for dim in dataArrayShape])) + dataArrayBuffer

        # Add message length to buffer
        outputBuffer = len(outputBuffer).to_bytes(4, self.ENDIANNESS) + outputBuffer

        return outputBuffer
    
    def BytesBufferToMultiTensor(self, inputDataBuffer: bytes) -> tuple[list[np.ndarray], list[tuple[int]], int]:
        """Function to convert a message containing multiple tensors in a buffer to a list of tensors. The buffer is expected to be in the following format:
        - 4 bytes: number of tensors (messages) (int)
        - for each tensor:
            - 4 bytes: message length (int)
            - 4 bytes: number of dimensions (int)
            - 4 bytes per dimension: shape of tensor (int)
            - remaining bytes: flattened tensor data (float32), column-major order
        Each tensor message is stacked in the buffer one after the other.
        
        Args:
            inputDataBuffer (bytes): Input bytes buffer

        Returns:
            tuple[list[np.ndarray], list[tuple[int]], int]: Tuple containing the list of tensors, their shapes and the number of tensors
        """
        # Get number of tensors
        numOfTensors = int.from_bytes(inputDataBuffer[:4], self.ENDIANNESS)  

        # Initialize list to store tensors
        dataArray = []
        dataArrayShape = []

        # Construct extraction ptrs
        ptrStart = 4 # First data message starts at byte 4 (after number of tensors)
    
        for idx in range(numOfTensors):
            
            # Get length of tensor message
            tensorMessageLength = int.from_bytes(inputDataBuffer[ptrStart:ptrStart+4], self.ENDIANNESS)

            # Extract sub-message from buffer
            subTensorMessage = inputDataBuffer[(ptrStart + 4) : (ptrStart + 4) + tensorMessageLength]

            # Call function to convert each tensor message to tensor
            tensor, tensorShape = self.BytesBufferToTensor(subTensorMessage)

            # Append data to list
            dataArray.append(tensor)
            dataArrayShape.append(tensorShape)

            # Update buffer ptr for next tensor message
            ptrStart += ptrStart + 4 + tensorMessageLength 
                                   
        return dataArray, dataArrayShape, numOfTensors


    def MultiTensorToBytesBuffer(self, processedData: Union[list, dict, tuple]) -> bytes:
        """Function to convert multiple tensors in a python container to multiple buffer messages of tensor convention:
        - 4 bytes: number of tensors (messages) (int)
        - for each tensor:
            - 4 bytes: message length (int)
            - 4 bytes: number of dimensions (int)
            - 4 bytes per dimension: shape of tensor (int)
            - remaining bytes: flattened tensor data (float32), column-major order        

        Args:
            processedData (Union[list, dict, tuple]): Input data container
        
        Raises:
            TypeError: If input data container type is not recognized

        Returns:
            bytes: Output bytes buffer
        """

        # Process container according to type
        # Get size of container
        numOfTensors = len(processedData)
        print(f"Number of tensors to process: {numOfTensors}")

        if isinstance(processedData, [list, tuple]):

            # Convert each tensor to buffer
            processedDataBufferList = [self.TensorToBytesBuffer(tensor) for tensor in processedData]
            # Concatenate all buffers
            outputBuffer = b''.join(*processedDataBufferList)

        elif isinstance(processedData, dict):
 
            # Convert each tensor to buffer
            processedDataBufferList = [self.TensorToBytesBuffer(tensor) for tensor in processedData.values()]
            # Concatenate all buffers
            outputBuffer = b''.join(*processedDataBufferList)
        else:
            raise TypeError('Input data container type not recognized. Please provide a list, tuple or dict.')
        
        return outputBuffer
        




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

# Dummy processing function for testing
def dummy_processing_function(data, num_batches):
    return data * 2

# Test DataProcessor class

def test_data_processor():
    processor = DataProcessor(
        dummy_processing_function, inputTargetType=np.float32, BufferSizeInBytes=1024)
    input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    output_data = processor.process(input_data)
    expected_output = (
        np.array([1.0, 2.0, 3.0], dtype=np.float32) * 2).tobytes()
    assert output_data == expected_output


def test_tcp_server():
    HOST, PORT = "localhost", 9999

    # Create a DataProcessor instance
    processor = DataProcessor(
        dummy_processing_function, inputTargetType=np.float32, BufferSizeInBytes=1024)

    # Create and start the server in a separate thread
    server = pytcp_server((HOST, PORT), pytcp_requestHandler, processor)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # Create a client socket to connect to the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))

        # Send the length of the data
        input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
        client.sendall(len(input_data).to_bytes(4, 'little'))

        # Send the actual data
        client.sendall(input_data)

        # Receive the length of the processed data
        data_length = int.from_bytes(client.recv(4), 'little')

        # Receive the processed data
        processed_data = client.recv(data_length)

        expected_output = (
            np.array([1.0, 2.0, 3.0], dtype=np.float32) * 2).tobytes()
        assert processed_data == expected_output


if __name__ == "__main__":
    test_data_processor()   
    test_tcp_server()
