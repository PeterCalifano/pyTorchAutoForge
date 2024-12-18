import numpy as np

# Custom imports
from pyTorchAutoForge.api.tcp import DataProcessor, pytcp_server, pytcp_requestHandler, ProcessingMode
import threading

# MAIN SCRIPT
def main():
    print('\n\n----------------------------------- RUNNING: testStartServer.py -----------------------------------\n')
    
    # %% TCP SERVER INITIALIZATION
    HOST1, PORT1 = "127.0.0.1", 50000 # Define host and port for the first server
    HOST2, PORT2 = "127.0.0.1", 50001 # Define host and port for the second server

    def dummy_function(inputData):
        return inputData

    # Define DataProcessor object for RequestHandler
    dataProcessorObj = DataProcessor(dummy_function, np.float32, 1024,
                                        ENDIANNESS='little', DYNAMIC_BUFFER_MODE=True,
                                        PRE_PROCESSING_MODE=ProcessingMode.TENSOR)
    
    dataProcessorObj_multi = DataProcessor(dummy_function, np.float32, 1024, 
                                                 ENDIANNESS='little', DYNAMIC_BUFFER_MODE=True, 
                                                 PRE_PROCESSING_MODE=ProcessingMode.MULTI_TENSOR)

    def start_server(host, port, dataProcessorObj):
        with pytcp_server((host, port), pytcp_requestHandler, dataProcessorObj, bindAndActivate=True) as server:
            try:
                print(f'\nServer initialized correctly on {host}:{port}. Set in "serve_forever" mode.')
                server.serve_forever()
            except KeyboardInterrupt:
                print(f"\nServer on {host}:{port} is gracefully shutting down =D.")
                server.shutdown()
                server.server_close()

    # Start two servers on separate threads
    thread1 = threading.Thread(target=start_server, args=( HOST1, PORT1, dataProcessorObj) )
    thread2 = threading.Thread(target=start_server, args=( HOST2, PORT2, dataProcessorObj_multi) )

    thread1.start()
    thread2.start()

    # Wait for the threads to finish processing
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    main()


