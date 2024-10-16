import os, subprocess, time

def StartMLflowUI(port: int = 8080):

    # Start MLflow UI
    os.system('mlflow ui --port ' + str(port))
    process = subprocess.Popen(['mlflow', 'ui', '--port ' + f'{port}', '&'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f'MLflow UI started with PID: {process.pid}, on port: {port}')
    time.sleep(1)  # Ensure the UI has started

    if process.poll() is None:
        print('MLflow UI is running OK.')
    else:
        raise RuntimeError('MLflow UI failed to start. Run stopped.')

    return process

# def GetMlflowRunByName() TODO