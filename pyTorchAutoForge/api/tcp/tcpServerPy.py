"""! Prototype TCP server script created by PeterC - 15-06-2024
ACHTUNG: sockerServer works in a http-like manner (open-handle-close client connections). his implementation is modified to use a while loop in the 
handle method, which processes an arbitracy number of the same requests from the same client. The server is kept busy in the meantime."""
# NOTE: the current implementation allows one request at a time, since it is thought for the evaluation of torch models in MATLAB.

# Python imports
from ast import Dict
from collections.abc import Callable
import socketserver
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Union
from enum import Enum
from torch import Tensor
import socket
import time
import threading
import sys
import msgpack
#import msgpack_numpy

# TODO by gdd:
# Modify handling --> Specialize each handle instead of passing the function to process.
# tcp server does not require specialization to store data processor.
# TODO: Fix typing errors in the code
# TODO: Add capability to transfer uint8 (for images) instead of floats (1/4 of bytes per pixel)
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
    MSG_PACK = 3

# %% Data processing function wrapper as generic interface in RequestHandler for TCP servers - PeterC - 15-06-2024


class DataProcessor():
    '''Data processing function wrapper as generic interface in RequestHandler for TCP servers. Input/output for numerical data: numpy.ndarray'''

    def __init__(self, processDataFcn: Callable, inputTargetType: Any = np.float32, BufferSizeInBytes: int = -1, ENDIANNESS: str = 'little',
                 DYNAMIC_BUFFER_MODE: bool = True, PRE_PROCESSING_MODE: ProcessingMode = ProcessingMode.MULTI_TENSOR) -> None:
        '''Constructor'''
        self.processDataFcn : Callable = processDataFcn
        self.inputTargetType = inputTargetType
        self.BufferSizeInBytes = BufferSizeInBytes
        self.DYNAMIC_BUFFER_MODE = DYNAMIC_BUFFER_MODE
        self.ENDIANNESS = ENDIANNESS
        self.PROCESSING_MODE = PRE_PROCESSING_MODE

    def process(self, inputDataBuffer: bytes) -> bytes:
        """Function to process input data buffer. It expects the buffer to be passed in the following format:
        - Without the initial 4 bytes of the length of the whole buffer, used by recv
        - Composition of messages in the "multi-tensor" format or a message directly convertible to a numpy array using frombuffer()
        This convention is assumed by deserialize() functionalities.

        Args:
            inputDataBuffer (bytes): _description_

        Returns:
            bytes: _description_
        """
        # Decode inputData
        decodedData, dataArrayShape = self.deserialize(inputDataBuffer)

        # Execute processing function
        # DEVNOTE TODO: replace by standard class method call
        processedData = self.processDataFcn(decodedData)
        # TODO: replace temporary input with a structured type like a dict to avoid multiple inputs and keep the function interface generic --> better to define a data class?

        return self.serialize(processedData)

    def deserialize(self, inputDataBuffer: bytes) -> tuple[np.ndarray | list[np.ndarray] | dict, tuple[int]]:
        '''Data conversion function from raw bytes stream to specified target numpy type with specified shape'''
        if not isinstance(inputDataBuffer, self.inputTargetType):

            if self.PROCESSING_MODE == ProcessingMode.TENSOR:
                # Convert input data to tensor shape
                [dataStruct, dataStructShape] = self.BytesBufferToTensor(
                    inputDataBuffer[4:])
                print(f"Received tensor of shape:\t{dataStructShape}")

            elif self.PROCESSING_MODE == ProcessingMode.MULTI_TENSOR:

                # Convert input data to multi-tensor list
                [dataStruct, dataStructShape, numOfTensors] = self.BytesBufferToMultiTensor(
                    inputDataBuffer)
                print(f"Received list of tensors of length:\t{numOfTensors}")

            elif self.PROCESSING_MODE == ProcessingMode.MSG_PACK:

                # Call msg_pack method for decoding
                dataStruct = self.BytesBufferToMsgPack(inputDataBuffer)

                if 'shape' in dataStruct.keys():
                    dataStructShape = dataStruct['shape']
                else:
                    dataStructShape = None

            else:
                # dataArray = np.array(np.frombuffer(inputDataBuffer[4:], dtype=self.inputTargetType), dtype=self.inputTargetType)
                dataArrayBuffer = inputDataBuffer
                try:
                    dataStruct = np.array(np.frombuffer(
                        dataArrayBuffer, dtype=self.inputTargetType), dtype=self.inputTargetType)
                    dataStructShape = dataStruct.shape

                except TypeError as errMsg:
                    print('Data conversion from raw data array to specified target type {targetType} failed with error: {errMsg}\n'.format(
                        targetType=self.inputTargetType, errMsg=str(errMsg)))
            return dataStruct, dataStructShape
        else:
            return inputDataBuffer, inputDataBuffer.shape

    def serialize(self, processedData):
        '''Data conversion function from numpy array to raw bytes stream'''
        if self.PROCESSING_MODE == ProcessingMode.TENSOR:
            # Convert processed data to tensor-convention buffer
            processedData = self.TensorToBytesBuffer(processedData)
            return processedData

        elif self.PROCESSING_MODE == ProcessingMode.MULTI_TENSOR:

            # Convert processed data multi-tensor to tensor-convention buffer
            processedData = self.MultiTensorToBytesBuffer(processedData)
            return processedData

        elif self.PROCESSING_MODE == ProcessingMode.MSG_PACK:

            # Call msg_pack method for encoding
            processedData = self.MsgPackToBytesBuffer(processedData)
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
        dataArrayShape = tuple(int.from_bytes(
            inputDataBuffer[4+4*(idx):8+4*(idx)], self.ENDIANNESS) for idx in range(numOfDims))

        # Convert buffer to numpy array with specified shape
        dataArray = np.array(np.frombuffer(
            inputDataBuffer[8+4*(numOfDims-1):], dtype=self.inputTargetType), dtype=self.inputTargetType).reshape(dataArrayShape, order='F')

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
        outputBuffer = numOfDims.to_bytes(4, self.ENDIANNESS) + (b''.join(
            [dim.to_bytes(4, self.ENDIANNESS) for dim in dataArrayShape])) + dataArrayBuffer

        # Add message length to buffer
        outputBuffer = len(outputBuffer).to_bytes(
            4, self.ENDIANNESS) + outputBuffer

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
        # TBC: inputDataBuffer may be provided to deserialize function without the first 4 bytes (message length)
        numOfTensors = int.from_bytes(inputDataBuffer[:4], self.ENDIANNESS)

        # Initialize list to store tensors
        dataArray = []
        dataArrayShape = []

        # Construct extraction ptrs
        # First data message starts at byte 4 (after number of tensors)
        ptrStart = 4
        __SIZE_OF_FLOAT32__ = 4  # Size of float32 in bytes

        for idx in range(numOfTensors):

            # Get length of tensor message
            tensorMessageLength = int.from_bytes(
                inputDataBuffer[ptrStart:ptrStart+4], self.ENDIANNESS)  # In bytes

            print(
                f"Processing Tensor message of length: {tensorMessageLength}")
            # Extract sub-message from buffer
            # Extract sub-message in bytes
            subTensorMessage = inputDataBuffer[ptrStart +
                                               4:(ptrStart + 4) + tensorMessageLength]

            # Call function to convert each tensor message to tensor
            tensor, tensorShape = self.BytesBufferToTensor(subTensorMessage)

            # Append data to list
            dataArray.append(tensor)
            dataArrayShape.append(tensorShape)

            # Update buffer ptr for next tensor message
            ptrStart = (ptrStart + 4) + tensorMessageLength

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

        # Automatic encapsulation if instance is single tensor/array
        if isinstance(processedData, Union[np.ndarray, Tensor]):
            processedData = [processedData]  # Convert to list

        # Process container according to type
        # Get size of container

        numOfTensors = len(processedData)
        print(f"Number of tensors to process: {numOfTensors}")

        if isinstance(processedData, Union[list, tuple]):

            # Convert each tensor to buffer
            processedDataBufferList = [self.TensorToBytesBuffer(
                tensor) for tensor in processedData]

            # Concatenate all buffers
            # outputBuffer = b''.join(processedDataBufferList)

            outputBuffer = numOfTensors.to_bytes(
                4, self.ENDIANNESS) + b''.join(processedDataBufferList)

        elif isinstance(processedData, dict):

            # Convert each tensor to buffer
            processedDataBufferList = [self.TensorToBytesBuffer(
                tensor) for tensor in processedData.values()]
            # Concatenate all buffers
            outputBuffer = numOfTensors.to_bytes(
                4, self.ENDIANNESS) + b''.join(processedDataBufferList)

        else:
            raise TypeError(
                'Input data container type not recognized. Please provide a list, tuple or dict.')

        return outputBuffer

    def MsgPackToBytesBuffer(self, processedData: dict | np.ndarray | Tensor ) -> bytes:
        """Function to convert a dictionary to a message pack buffer. The dictionary is expected to contain the following keys:
        - 'data': numpy array data
        - 'shape': tuple of data shape

        Args:
            processedData (dict): Input data dictionary # TBC input data

        Returns:
            bytes: Output bytes buffer
        """

        if not isinstance(processedData, dict) and isinstance(processedData, (np.ndarray, Tensor)):
            # Attempt encapsulation if not a dictionary
            processedData = {'data': processedData, 'shape': processedData.shape}

        elif not isinstance(processedData, dict):
            raise TypeError(
                'Input data must be a dictionary object, a numpy.ndarray or a torch.tensor.')

        # Perform type checking for entries of dictionary
        for key, value in processedData.items():
            if isinstance(value, (np.ndarray | Tensor)):
                # If numpy arrays or torch tensors in dict, convert to lists
                processedData[key] = value.tolist()
            elif not(isinstance(value, (list, tuple, str, int, float))):
                # If not another type known to msgpack, raise error
                raise TypeError('Input data type must be known to msgpack!')
            
            elif isinstance(value, dict):
                raise NotImplementedError('Throwing error to avoid incorrect serialization: input item type is dict, but recursive type checking is not implemented yet.')

        # Convert input data to message pack buffer
        outputBuffer = msgpack.packb(processedData)

        return outputBuffer

    def BytesBufferToMsgPack(self, inputDataBuffer: bytes) -> dict:
        """Function to convert a message pack buffer to a dictionary. The buffer is expected to be in the message pack format.

        Args:
            inputDataBuffer (bytes): Input bytes buffer

        Returns:
            dict: Output data dictionary
        """

        # Convert message pack buffer to dictionary
        outputData = msgpack.unpackb(inputDataBuffer)

        # Convert all lists to numpy arrays
        for key, value in outputData.items():

            if isinstance(value, (list | tuple)):
                outputData[key] = np.array(value)

        return outputData


# %% Request handler class - PeterC + GPT4o- 15-06-2024
class pytcp_requestHandler(socketserver.BaseRequestHandler):
    '''Request Handler class for tcp server'''

    def __init__(self, request, client_address, server, DataProcessor: DataProcessor, ENDIANNESS: str = 'little'):
        ''''Constructor'''
        self.DataProcessor = DataProcessor  # Initialize DataProcessing object for handle
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
        '''Function handling request from client. Automatically reads and adds first 4 bytes of message as message length (outside serialize/deserialize functions)'''
        print(f"Handling request from client: {self.client_address}")
        try:
            while True:
                # Read the length of the data (4 bytes) specified by the client
                bufferSizeFromClient = self.request.recv(4)
                if not bufferSizeFromClient:
                    break
                # NOTE: MATLAB writes as LITTLE endian
                bufferSize = int.from_bytes(
                    bufferSizeFromClient, self.ENDIANNESS)

                # Print received length bytes for debugging a
                print(f"Received length bytes: {bufferSizeFromClient}",
                      ", ", f"Interpreted length: {bufferSize}")

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
                if not (self.DYNAMIC_BUFFER_MODE):
                    print("Expected data buffer size from client:",
                          bufferSizeExpected, "bytes")
                    if not (len(dataBuffer) == bufferSizeExpected):
                        raise BufferError('Data buffer size does not match buffer size by Data Processor! Received message contains {nBytesReceived}'.format(
                            nBytesReceived=len(dataBuffer)))
                    else:
                        print(
                            'Message size matches expected size. Calling data processor...')

                # Data processing handling: move the data to DataProcessor and process according to specified function
                outputDataSerialized = self.DataProcessor.process(dataBuffer)
                # For strings: outputDataSerialized = ("Acknowledge message. Array was received!").serialize('utf-8')

                # Get size of serialized output data
                outputDataSizeInBytes = len(outputDataSerialized)
                print('Sending total number of bytes to client:',
                      outputDataSizeInBytes+4)

                # Send the length of the processed data
                self.request.sendall(
                    outputDataSizeInBytes.to_bytes(4, self.ENDIANNESS))

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

    def __init__(self, serverAddress: tuple[str | bytes | bytearray, int], RequestHandlerClass: pytcp_requestHandler, DataProcessor: DataProcessor, bindAndActivate: bool = True) -> None:
        '''Constructor for custom tcp server'''
        self.DataProcessor = DataProcessor  # Initialize DataProcessing object for handle
        super().__init__(serverAddress, RequestHandlerClass, bindAndActivate)
        print('Server opened on (HOST, PORT): (',
              serverAddress[0], ', ', serverAddress[1], ')')

    def finish_request(self, request, client_address) -> None:
        '''Function evaluating Request Handler'''
        self.RequestHandlerClass(
            request, client_address, self, self.DataProcessor)

# %% TEST CODES
# Dummy processing function for testing


def dummy_processing_function(data):
    if isinstance(data, Union[list, tuple]):
        # Achtung: input "data" is a list --> what happens is duplication
        return [2.0*data_ for data_ in data]
    elif isinstance(data, np.ndarray):
        return 2.0*data
    else:
        raise TypeError('Input data must be a list or numpy array.')

# Test DataProcessor class


def dummy_processing_function_msgpack(dataDict: dict):

    if isinstance(dataDict, dict):
        print('Data received: ', dataDict)
        print('Shape of data: ', dataDict['shape'])
        return {'data': dataDict['data'], 'shape': dataDict['shape']} # Return back the data
    else:
        raise TypeError('Input data must be a dict')




def packAsBuffer_and_process_wrapper(input_data, processor):

    if isinstance(input_data, Union[np.ndarray, Tensor]):
        # Convert input data to bytes buffer
        input_data_bytes = processor.TensorToBytesBuffer(input_data)
    elif isinstance(input_data, Union[list, dict, tuple]):
        # Convert input data to bytes buffer using multi tensor mode
        input_data_bytes = processor.MultiTensorToBytesBuffer(input_data)

    # Process the input data
    output_data = processor.process(input_data_bytes)

    return output_data


def test_data_processor_tensor_mode_1D():
    processor = DataProcessor(dummy_processing_function, inputTargetType=np.float32, BufferSizeInBytes=-1, ENDIANNESS='little',
                              DYNAMIC_BUFFER_MODE=True, PRE_PROCESSING_MODE=ProcessingMode.TENSOR)

    # Create dummy input data (1D tensor)
    input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Call process method of DataProcessor
    output_data = packAsBuffer_and_process_wrapper(input_data, processor)

    # Check bytes stream according to how it is constructed
    expected_output = (len(input_data.shape)).to_bytes(4, 'little') + b''.join([shape.to_bytes(
        4, 'little') for shape in input_data.shape]) + (input_data * 2).tobytes()

    assert output_data[4:] == expected_output, 'Processed data does not match expected output!'
    print('\n1D tensor processing test passed!\n')


def test_data_processor_tensor_mode_2D():
    processor = DataProcessor(dummy_processing_function, inputTargetType=np.float32, BufferSizeInBytes=-1, ENDIANNESS='little',
                              DYNAMIC_BUFFER_MODE=True, PRE_PROCESSING_MODE=ProcessingMode.TENSOR)

    # Create dummy input data (2D tensor)
    input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    output_data = packAsBuffer_and_process_wrapper(input_data, processor)
    expected_output = processor.TensorToBytesBuffer(2.0*input_data)

    # Check message length
    assert output_data[:4] == expected_output[:4], 'Message length does not match!'
    # Check processed data
    # If multi-tensor --> first data starts at byte 20 (numOfTensors, 1st msg length, numOfDims, shape_for_each_dim), else starts at byte 12
    assert output_data[4:] == expected_output[4:], 'Message data do not match'
    print('\n2D tensor processing test passed!\n')


def test_data_processor_tensor_mode_4D():
    processor = DataProcessor(dummy_processing_function, inputTargetType=np.float32, BufferSizeInBytes=-1, ENDIANNESS='little',
                              DYNAMIC_BUFFER_MODE=True, PRE_PROCESSING_MODE=ProcessingMode.TENSOR)

    # Create dummy large size input data (4D tensor)
    input_data = np.random.rand(2, 3, 4, 5).astype(np.float32)
    output_data = packAsBuffer_and_process_wrapper(input_data, processor)

    expected_output = processor.TensorToBytesBuffer(2.0*input_data)

    # If multi-tensor --> first data starts at byte 20 (numOfTensors, 1st msg length, numOfDims, shape_for_each_dim), else starts at byte 12
    assert output_data[:20] == expected_output[:
                                               20], 'Message length and number of dimensions do not match!'
    assert output_data[20:] == expected_output[20:
                                               ], 'Processed data does not match expected output!'
    print('\n4D tensor processing test passed!\n')


def test_data_processor_multi_tensor_mode_2D():
    processor = DataProcessor(dummy_processing_function, inputTargetType=np.float32, BufferSizeInBytes=-1, ENDIANNESS='little',
                              DYNAMIC_BUFFER_MODE=True, PRE_PROCESSING_MODE=ProcessingMode.MULTI_TENSOR)

    # Define image-like arrays with a circle in the middle
    # Black background with white circle
    image1 = np.zeros((100, 100), dtype=np.float32)
    # Create a white circle in the middle
    for i in range(100):
        for j in range(100):
            if (i-50)**2 + (j-50)**2 <= 25**2:
                image1[i, j] = 1.0

    # White background with black circle
    image2 = np.ones((100, 100), dtype=np.float32)
    # Create a black circle in the middle
    for i in range(100):
        for j in range(100):
            if (i-50)**2 + (j-50)**2 <= 25**2:
                image2[i, j] = 0.0

    # Encode data to buffer and process
    input_data = [image1, image2]
    output_data = packAsBuffer_and_process_wrapper(input_data, processor)

    # Check bytes stream according to how it is constructed
    expected_output_num_msg = (len(input_data)).to_bytes(4, 'little')
    msg1_bytes = processor.TensorToBytesBuffer(2.0 * image1)
    msg2_bytes = processor.TensorToBytesBuffer(2.0 * image2)

    # Get length of each message
    msg1_length = len(msg1_bytes)
    msg2_length = len(msg2_bytes)

    # Check expected output sizes and msg lengths
    assert output_data[:4] == expected_output_num_msg[:
                                                      4], 'Number of messages does not match!'
    assert output_data[4:8] == msg1_bytes[:4], 'Message 1 length does not match!'
    assert output_data[4 + msg1_length: 4 + msg1_length +
                       4] == msg2_bytes[:4], 'Message 2 length does not match!'

    # Check data
    assert output_data[8: msg1_length +
                       4] == msg1_bytes[4:], 'Message 1 shape sizes and data does not match!'
    assert output_data[msg1_length + 4 +
                       4:] == msg2_bytes[4:], 'Message 2 shape sizes and data does not match!'

    print('\nMulti-tensor processing test passed!\n')


def BuildMessage(input_data, processor):
    if isinstance(input_data, np.ndarray | Tensor):
        # Convert input data to bytes buffer
        input_data_bytes = processor.TensorToBytesBuffer(input_data)
    elif isinstance(input_data, list | dict | tuple):
        # Convert input data to bytes buffer using multi tensor mode
        input_data_bytes = processor.MultiTensorToBytesBuffer(
            input_data)

    # Add message length to buffer
    input_data_bytes = len(input_data_bytes).to_bytes(
        4, 'little') + input_data_bytes

    return input_data_bytes


def test_tcp_server_tensor_mode():
    HOST, PORT = "localhost", 9999

    # Create a DataProcessor instance
    processor_tensor = DataProcessor(
        dummy_processing_function, inputTargetType=np.float32, BufferSizeInBytes=1024, ENDIANNESS='little', DYNAMIC_BUFFER_MODE=True, PRE_PROCESSING_MODE=ProcessingMode.TENSOR)

    # Create and start the server in a separate thread
    server = pytcp_server((HOST, PORT), pytcp_requestHandler, processor_tensor)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # Create a client socket to connect to the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))

        # Build test messages
        input_data_1D = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        input_data_2D = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        input_data_4D = np.random.rand(2, 3, 4, 5).astype(np.float32)

        # TEST MESSAGE 1
        # Send message 1 to server
        client.sendall(BuildMessage(input_data_1D, processor_tensor))
        time.sleep(0.2)

        try:
            # Receive processed data from server
            data_length = int.from_bytes(client.recv(4), 'little')
            processed_data = client.recv(data_length)

            # Assert processed data
            expected_output = processor_tensor.TensorToBytesBuffer(
                2.0*input_data_1D)
            assert processed_data == expected_output

        except ConnectionResetError as err:
            print('Connection reset error occurred: ', err, 'Test skipped.')

        # TEST MESSAGE 2
        # Send message 2 to server
        client.sendall(BuildMessage(input_data_2D, processor_tensor))
        time.sleep(0.2)

        try:

            # Receive processed data from server
            data_length = int.from_bytes(client.recv(4), 'little')
            processed_data = client.recv(data_length)

            # Assert processed data
            expected_output = processor_tensor.TensorToBytesBuffer(
                2.0*input_data_2D)
            assert processed_data == expected_output

        except ConnectionResetError as err:
            print('Connection reset error occurred: ', err, 'Test skipped.')

        # TEST MESSAGE 3
        # Send message 3 to server
        client.sendall(BuildMessage(input_data_4D, processor_tensor))
        time.sleep(0.2)

        try:
            # Receive processed data from server
            data_length = int.from_bytes(client.recv(4), 'little')
            processed_data = client.recv(data_length)

        # Assert processed data
            expected_output = processor_tensor.TensorToBytesBuffer(
                2.0*input_data_4D)
            assert processed_data == expected_output

        except ConnectionResetError as err:
            print('Connection reset error occurred: ', err, 'Test skipped.')

        time.sleep(1)
        print('\nTensor processing test passed!\n')

    server.shutdown()
    server.server_close()


def test_tcp_server_multi_tensor_mode():
    HOST, PORT = "localhost", 9998

    # Create a DataProcessor instance
    processor_multitensor = DataProcessor(
        dummy_processing_function, inputTargetType=np.float32, BufferSizeInBytes=1024, ENDIANNESS='little', DYNAMIC_BUFFER_MODE=True, PRE_PROCESSING_MODE=ProcessingMode.MULTI_TENSOR)

    # Create and start the server in a separate thread
    server = pytcp_server(
        (HOST, PORT), pytcp_requestHandler, processor_multitensor)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # Create a client socket to connect to the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))

        # Build test messages
        input_data_multi_tensor = [
            np.zeros((100, 100), dtype=np.float32), np.ones((100, 100), dtype=np.float32)]

        # TEST MESSAGE 1
        # Send message 1 to server
        client.sendall(BuildMessage(
            input_data_multi_tensor, processor_multitensor))
        time.sleep(0.5)

        # Receive processed data from server
        data_length = int.from_bytes(client.recv(4), 'little')
        processed_data = client.recv(data_length)

        # Assert processed
        expected_output_num_msg = (
            len(input_data_multi_tensor)).to_bytes(4, 'little')

        try:
            # Check expected output sizes and msg lengths
            msg1_bytes = processor_multitensor.TensorToBytesBuffer(
                2.0 * input_data_multi_tensor[0])
            assert processed_data[:4] == expected_output_num_msg[:
                                                                 4], 'Number of messages does not match!'
            assert processed_data[4:8] == msg1_bytes[:
                                                     4], 'Message 1 length does not match!'
            msg2_bytes = processor_multitensor.TensorToBytesBuffer(
                2.0 * input_data_multi_tensor[1])

            # Get length of each message
            msg1_length = len(msg1_bytes)

            # Check data
            assert processed_data[4 + msg1_length: 4 + msg1_length +
                                  4] == msg2_bytes[:4], 'Message 2 length does not match!'

            assert processed_data[8: msg1_length +
                                  4] == msg1_bytes[4:], 'Message 1 shape sizes and data does not match!'
            assert processed_data[msg1_length + 4 +
                                  4:] == msg2_bytes[4:], 'Message 2 shape sizes and data does not match!'

        except ConnectionResetError as err:
            print('Connection reset error occurred: ', err, 'Test skipped.')

        time.sleep(1)
        print('\nMulti-tensor processing test passed!\n')

    server.shutdown()
    server.server_close()


def test_tcp_server_msgpack():
    HOST, PORT = "localhost", 9998

    # Create a DataProcessor instance
    processor_msgpack = DataProcessor(
        dummy_processing_function_msgpack, inputTargetType=np.float32, BufferSizeInBytes=1024, ENDIANNESS='little', DYNAMIC_BUFFER_MODE=True, PRE_PROCESSING_MODE=ProcessingMode.MSG_PACK)

    # Create and start the server in a separate thread
    server = pytcp_server(
        (HOST, PORT), pytcp_requestHandler, processor_msgpack)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()


    # Create a client socket to connect to the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))

        # Build test messages
        example_image = np.zeros((100, 100), dtype=np.float32)

        # Create a white circle in the middle
        for i in range(100):
            for j in range(100):
                if (i-50)**2 + (j-50)**2 <= 25**2:
                    example_image[i, j] = 1.0

        # Create a dictionary with data and shape
        input_data_msgpack = {'data': example_image, 'shape': example_image.shape}

        # Create a message pack buffer
        input_data_msgpack_bytes = processor_msgpack.MsgPackToBytesBuffer(
            input_data_msgpack)

        # Send data to server
        client.sendall(len(input_data_msgpack_bytes).to_bytes(4, 'little') + input_data_msgpack_bytes)
        time.sleep(0.5)

        # Receive processed data from server
        data_length = int.from_bytes(client.recv(4), 'little')
        processed_data_bytes = client.recv(data_length)

        # Convert from msgpack to dictionary
        processed_data_dict = processor_msgpack.BytesBufferToMsgPack(
            processed_data_bytes)

        # Get data from dict
        processed_data = processed_data_dict['data']
        processed_shape = processed_data_dict['shape']

        # Show image in dict
        print('Processed data: ', processed_data.shape)
        
        import cv2 as ocv
        ocv.imshow('Processed image', processed_data)
        ocv.waitKey(2000)
        ocv.destroyAllWindows()

        # Assertions
        assert (processed_shape == example_image.shape).all(), 'Processed data shape does not match!'
        assert processed_data_bytes == input_data_msgpack_bytes, 'Processed data dict does not match input data dict!'

    server.shutdown()
    server.server_close()

if __name__ == "__main__":
    test_data_processor_tensor_mode_1D()
    test_data_processor_tensor_mode_2D()
    test_data_processor_tensor_mode_4D()
    test_data_processor_multi_tensor_mode_2D()
    test_tcp_server_tensor_mode()
    test_tcp_server_multi_tensor_mode()
    test_tcp_server_msgpack()

    print('All tests passed!')
