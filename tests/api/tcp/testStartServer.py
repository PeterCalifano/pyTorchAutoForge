import numpy as np

# Custom imports
from pyTorchAutoForge.api.tcp import DataProcessor, pytcp_server, pytcp_requestHandler, ProcessingMode

# MAIN SCRIPT
def main():
    print('\n\n----------------------------------- RUNNING: testStartServer.py -----------------------------------\n')
    
    # %% TCP SERVER INITIALIZATION
    HOST, PORT = "127.0.0.1", 50000 # Define host and port (random is ok)

    def dummy_function(inputData):
        return inputData

    # Define DataProcessor object for RequestHandler
    dataProcessorObj = DataProcessor(dummy_function, np.float32, 1024, 
                                                 ENDIANNESS='little', DYNAMIC_BUFFER_MODE=True, 
                                                 PRE_PROCESSING_MODE=ProcessingMode.TENSOR)

    # Initialize TCP server and keep it running
    with pytcp_server((HOST, PORT), pytcp_requestHandler, dataProcessorObj, bindAndActivate=True) as server:
        try:
            print('\nServer initialized correctly. Set in "serve_forever" mode.')
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer is gracefully shutting down =D.")
            server.shutdown()
            server.server_close()

if __name__ == "__main__":
    main()


