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
    :raises FileNotFoundError: Raised when the specified YAML file is not found.
    :return: An instance of the configuration class populated with the provided data.
    :rtype: BaseConfigClass
        Load configuration data from a YAML file and create an instance of the configuration class.
        :param path: The file path to the YAML configuration file.
        :type path: str
        :raises FileNotFoundError: If the specified YAML file does not exist.
        :return: An instance of the configuration class populated with the data from the YAML file.
        :rtype: T
        pass
        Create an instance of the configuration class from a dictionary.
        :param data: A dictionary containing configuration data.
        :type data: dict
        :return: An instance of the configuration class populated with the provided dictionary data.
        :rtype: T"
    """
    @classmethod
    def from_yaml(cls: type[T], path: str) -> T:

        # Check file is found
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file {path} not found.")
        
        with open(path, 'r') as f:
            data_payload = yaml.safe_load(f) # Try to load yaml content

        return cls.from_dict(data_payload) 

    @classmethod
    def from_dict(cls: type[T], data: dict) -> T:
        # Build config class from dict using dacite
        return from_dict(data_class=cls, data=data, config=Config(strict=True, strict_unions_match=True))
