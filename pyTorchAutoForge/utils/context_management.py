import sys, signal
#from inputimeout import TimeoutOccurred

DO_IMPORT = False

# NOTE: signal.alarm does not work on Windows!
# See https://stackoverflow.com/questions/8420422/python-windows-equivalent-of-sigalrm
class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException



