import subprocess, os, psutil, signal
from torch.utils.tensorboard import SummaryWriter # SummaryWriter from torch.utils.tensorboard


# %% TENSORBOARD functions - 04-06-2024
# Function to check if Tensorboard is running
def IsTensorboardRunning() -> bool:
    """Check if TensorBoard is already running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'tensorboard' in proc.info['cmdline']:
            # return proc.info['pid']
            return True
    return False

# Function to start TensorBoard process


def StartTensorboard(logDir: str, portNum: int = 6006) -> None:
    subprocess.Popen(['tensorboard', '--logdir', logDir,
                     '--host', '0.0.0.0', '--port', str(portNum)])

    # if not(IsTensorboardRunning):
    #    try:
    #        subprocess.Popen(['tensorboard', '--logdir', logDir, '--host', '0.0.0.0', '--port', str(portNum)])
    #        print('Tensorboard session successfully started using logDir:', logDir)
    #    except Exception as errMsg:
    #        print('Failed due to:', errMsg, '. Continuing without opening session.')
    # else:
    #    print('Tensorboard seems to be running in this session! Restarting with new directory...')
    # kill_tensorboard()
    # subprocess.Popen(['tensorboard', '--logdir', logDir, '--host', '0.0.0.0', '--port', '6006'])
    # print('Tensorboard session successfully started using logDir:', logDir)

# Function to stop TensorBoard process


def KillTensorboard():
    """Kill all running TensorBoard instances."""
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'tensorboard' in process.info['name']:
            if process.info['cmdline'] != None:
                for cmd in process.info['cmdline']:
                    if 'tensorboard' in cmd:
                        print(
                            f"Killing process {process.info['pid']}: {process.info['cmdline']}")
                        os.kill(process.info['pid'], signal.SIGTERM)

# Function to initialize Tensorboard session and writer


def ConfigTensorboardSession(logDir: str = './tensorboardLogs', portNum: int = 6006) -> SummaryWriter:

    print('Tensorboard logging directory:', logDir, ' Port number:', portNum)
    StartTensorboard(logDir, portNum)
    # Define writer # By default, this will write in a folder names "runs" in the directory of the main script. Else change providing path as first input.
    tensorBoardWriter = SummaryWriter(
        log_dir=logDir, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')

    # Return initialized writer
    return tensorBoardWriter
