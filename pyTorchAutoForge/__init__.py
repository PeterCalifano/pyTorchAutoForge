# Import scripts of the library at initialization
import os
import importlib
import importlib, os

# Removed modules
excluded_modules = ['tests', 'tensorboard']

def lazy_import(subpackage_name):
    module = None

    def load():
        nonlocal module
        if module is None:
            module = importlib.import_module(
                f'.{subpackage_name}', package=__name__)
        return module

    return load


def getModulesNames(excluded_dirs = []):
    # Get the path of the current package
    package_dir = os.path.dirname(__file__)

    # List all the sub-packages or modules, excluding items from the list
    subpackages = [
        name for name in os.listdir(package_dir)
        if os.path.isdir(os.path.join(package_dir, name)) and name not in excluded_modules
    ]


def lazy_import_subpackage(excluded_modules=[]):

    # For each sub-package or module, create a lazy loader
    for subpackage in getModulesNames(excluded_modules):
        if subpackage != '__pycache__':  # Exclude __pycache__ folder
            globals()[subpackage] = lazy_import(
                subpackage_name=f'{__name__}.{subpackage}')


if __name__ == '__main__':
    print('Initializing with all sub-packages and modules except:', excluded_modules)
    lazy_import_subpackage(excluded_modules)



