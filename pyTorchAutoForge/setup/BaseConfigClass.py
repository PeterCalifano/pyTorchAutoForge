#from dataclasses import field, dataclass
from dacite import from_dict, Config
import yaml
from typing import TypeVar
import os

# Define a type variable for the dataclass such that derived are enforced to inherit from BaseConfigClass
T = TypeVar('T', bound='BaseConfigClass')

class BaseConfigClass:
    """
    Base configuration class for loading and parsing configuration data.
    This class provides methods to load configuration data from a YAML file
    or a dictionary and convert it into a dataclass-based configuration object.

    Methods:
        from_yaml(cls, path: str) -> T:
            Load configuration data from a YAML file and create an instance of the configuration class.
            :param path: The file path to the YAML configuration file.
            :type path: str
            :raises FileNotFoundError: If the specified YAML file does not exist.
            :raises ValueError: If the YAML file is empty or has an invalid format.
            :return: An instance of the configuration class populated with the data from the YAML file.
            :rtype: T

        from_dict(cls, data: dict) -> T:
            Create an instance of the configuration class from a dictionary.
            :param data: A dictionary containing configuration data.
            :type data: dict
            :return: An instance of the configuration class populated with the provided dictionary data.
            :rtype: T
    """
    @classmethod
    def from_yaml(cls: type[T], path: str) -> T:

        # Check file is found
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file {path} not found.")
        
        with open(path, 'r') as file:
            data_payload = yaml.safe_load(file)  # Try to load yaml content

        # Assert if file is empty
        if data_payload is None:
            raise ValueError(f"Config file {path} is empty or invalid YAML format.")
        
        return cls.from_dict(data_payload) 

    @classmethod
    def from_dict(cls: type[T], data: dict) -> T:
        # Build config class from dict using dacite
        return from_dict(data_class=cls, data=data, config=Config(strict=True, strict_unions_match=False))
