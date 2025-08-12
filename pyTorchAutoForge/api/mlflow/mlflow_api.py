from __future__ import annotations

import os
import subprocess
import time
import sys
import mlflow
from dataclasses import is_dataclass, MISSING, field, dataclass, fields
from pathlib import Path
from typing import TypeVar, Type
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


def _serialize_dataclass(dataclass_obj: object) -> dict:
    if not (is_dataclass(dataclass_obj) and not isinstance(dataclass_obj, type)):
        raise TypeError(
            f"Expected a dataclass instance, got {type(dataclass_obj)}")
    
    # If a custom serializer exists, use it
    to_dict = getattr(dataclass_obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    # Else return raw fields
    data_dict = {}
    for f in fields(dataclass_obj):
        # for init=False fields that might not exist, skip if truly missing
        val = getattr(dataclass_obj, f.name, MISSING)
        if val is not MISSING:
            data_dict[f.name] = val
    return data_dict

# %% Auxiliary functions for mlflow tracking API


def StartMLflowUI(port: int = 8080):
    """
    StartMLflowUI _summary_

    _extended_summary_

    :param port: _description_, defaults to 8080
    :type port: int, optional
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: _type_
    """

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


def RecursiveLogParamsInDict(log_dict: dict | object, 
                             unwrap_depth: int = 1):
    """
    RecursiveLogParamsInDict _summary_

    _extended_summary_

    :param log_dict: _description_
    :type log_dict: dict
    :param unwrap_depth: _description_, defaults to 1
    :type unwrap_depth: int, optional
    """

    if is_dataclass(log_dict):
        # Serialize to dictionary
        log_dict = _serialize_dataclass(log_dict)

    if not isinstance(log_dict, dict):
        raise TypeError(
            f"Expected a dictionary or dataclass, got {type(log_dict)}")

    # Iterate through values in dict
    for key, value in log_dict.items():

        if is_dataclass(value):
            # Serialize to dictionary
            value = _serialize_dataclass(value)

        if isinstance(value, dict) and unwrap_depth > 0:
            # Recurse nested fields up to unwrap_depth
            RecursiveLogParamsInDict(value, unwrap_depth - 1)

        else:
            try:
                # Log parameter if scalar
                if not isinstance(value, dict):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_params(value, synchronous=True)
            except Exception as e:
                print("\033[31m" + f"Error logging parameter '{key}'. Handled by skipping entry." + "\033[0m")


def SetupMlflowTrackingSession(experiment_name: str,
                               database_root_dir: str | Path | None = None,
                               use_mlflow_remote_server: bool = False,
                               local_port: int = 7500,
                               DEBUG_MODE: bool = False,
                               database_filename: str = "mlflow_database"):

    if not isinstance(database_root_dir, (str, Path)) and not database_root_dir is None:
        raise TypeError("\033[31m" + f"database_root_dir must be a string, Path, or None. Got {type(database_root_dir)}" + "\033[0m")

    # Set up MLflow to use a SQLite backend database (standard general db)
    if database_root_dir is None:
        database_root_dir = os.getenv("SCRATCH")
        if database_root_dir is not None:
            print("\033[38;5;208mWARNING no root dir for database provided. Found SCRATCH env. variable will be used: " +
                  database_root_dir + "\033[0m")

            # Define a directory under SCRATCH to hold the SQLite file
            database_root_dir = os.path.join(database_root_dir)

        else:
            print("\033[38;5;208mWARNING: no root dir for database provided and SCRATCH env. variable is not set. Using current working directory...\033[0m")
            database_root_dir = "."
    else:
        database_root_dir = str(database_root_dir)

    database_root_dir = os.path.join(
        database_root_dir, "mlflow-storage", "mlflow-logs")
    os.makedirs(database_root_dir, exist_ok=True)

    # Path to the sqlite database
    sqlite_path = os.path.join(database_root_dir, f"{database_filename}.db")

    # MLflow database URI must start with sqlite:///
    tracking_uri = f"sqlite:///{sqlite_path}"

    if use_mlflow_remote_server and local_port is not None:
        tracking_uri = f"http://127.0.0.1:{local_port}"

    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    print(f"--- MLflow tracking URI set to: {tracking_uri}")

    # Create or select experiment
    exp_name = experiment_name
    if DEBUG_MODE:
        exp_name = "debug_session"

    # Set experiment and get experiment object
    experiment_obj = mlflow.set_experiment(exp_name)

    return experiment_obj, tracking_uri


def CleanExperimentRuns(experiment_id: str, 
                        include_all: bool = True) -> int:
    """
    Remove all runs from an experiment without deleting the experiment.

    Args:
        experiment_id: The MLflow experiment ID.
        include_all: If True, includes all runs (both active and deleted).

    Returns:
        The number of runs marked as deleted.
    """
    client: MlflowClient = MlflowClient()
    view_type = ViewType.ALL if include_all else ViewType.ACTIVE_ONLY

    runs = client.search_runs([experiment_id], 
                              filter_string="", 
                              run_view_type=view_type, 
                              max_results=50000)

    for r in runs:
        client.delete_run(r.info.run_id)  # Delete entry in experiment

    return len(runs)


# %% Test auxiliary functions
def _setup_mlflow_dbase_for_tests():

    archive_location = os.getenv("ARCHIVE")
    assert (archive_location is not None), "ARCHIVE environment variable not set"

    # Setup database and experiment
    experiment_obj, track_uri = SetupMlflowTrackingSession("debug_session",
                                                           archive_location,
                                                           False)
    # Clean up all runs in experiment
    #CleanExperimentRuns(experiment_obj.experiment_id, True)

    return experiment_obj, track_uri


@dataclass
class DummyDataClass:
    param1: int = 0
    param2: str = "default"
    param3: dict = field(default_factory=dict)


# %% Test functions
def test_mlflow_setup():
    _setup_mlflow_dbase_for_tests()


def test_mlflow_recursive_dict_logging_depth0():
    experiment_obj, track_uri = _setup_mlflow_dbase_for_tests()

    # Test logging of parameters
    test_params = {
        "param1": 5,
        "param2": "test",
        "param3": {
            "subparam1": 10,
            "subparam2": "test_sub"
        }
    }

    mlflow.start_run(experiment_id=experiment_obj.experiment_id, run_name="test_mlflow_recursive_dict_logging_depth0")

    # Test with input dictionary
    RecursiveLogParamsInDict(test_params, unwrap_depth=0)
    mlflow.end_run()


def test_mlflow_recursive_dict_logging_depth1():
    experiment_obj, track_uri = _setup_mlflow_dbase_for_tests()

    # Test logging of parameters
    test_params = {
        "param1": 5,
        "param2": "test",
        "param3": {
            "subparam1": 10,
            "subparam2": "test_sub"
        }
    }

    mlflow.start_run(experiment_id=experiment_obj.experiment_id, run_name="test_mlflow_recursive_dict_logging_depth1")
    # Test with input dictionary
    RecursiveLogParamsInDict(test_params, unwrap_depth=1)
    mlflow.end_run()


def test_mlflow_recursive_dataclass_logging_with_depth1():
    experiment_obj, track_uri = _setup_mlflow_dbase_for_tests()

    # Test logging of parameters
    test_params = {
        "param1": 5,
        "param2": "test",
        "param3": {
            "subparam1": 10,
            "subparam2": "test_sub"
        }
    }

    data_class_to_log = DummyDataClass(**test_params)

    mlflow.start_run(experiment_id=experiment_obj.experiment_id,
                     run_name="test_mlflow_recursive_dataclass_logging_with_depth1")
    
    # Test with input dictionary
    RecursiveLogParamsInDict(data_class_to_log, unwrap_depth=1)
    mlflow.end_run()


def test_mlflow_recursive_dict_nested_dataclass_logging_with_depth1():
    experiment_obj, track_uri = _setup_mlflow_dbase_for_tests()

    # Test logging of parameters
    test_params = {
        "param1": 5,
        "param2": "test",
        "param3": {
            "subparam1": 10,
            "subparam2": "test_sub"
        }
    }

    dict_to_log = {"my_dataclass": DummyDataClass(
        **test_params), "another_key": "another_value"}

    mlflow.start_run(experiment_id=experiment_obj.experiment_id,
                     run_name="test_mlflow_recursive_dict_nested_dataclass_logging_with_depth1")
    
    RecursiveLogParamsInDict(dict_to_log, unwrap_depth=1)
    mlflow.end_run()


# Manual execution for development
if __name__ == '__main__':

    test_mlflow_recursive_dict_logging_depth0()
    test_mlflow_recursive_dict_logging_depth1()
    test_mlflow_recursive_dataclass_logging_with_depth1()
    test_mlflow_recursive_dict_nested_dataclass_logging_with_depth1()
