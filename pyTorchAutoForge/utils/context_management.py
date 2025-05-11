import sys, signal
#from inputimeout import TimeoutOccurred

DO_IMPORT = False

# NOTE: signal.alarm does not work on Windows!
# See https://stackoverflow.com/questions/8420422/python-windows-equivalent-of-sigalrm
class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException


# TODO implement conveniency context management for PTAF facilities
# Example usage
if DO_IMPORT:
    from contextlib import contextmanager
    import mlflow
    @contextmanager
    def training_loop_context():
        try:
            yield
        except KeyboardInterrupt:
            if mlflow_logging: # if mlflow run active no None
                mlflow.end_run('KILLED')
            raise  # or sys.exit()

    # usage:
    #with interruptible_run():
    #    train_model()

