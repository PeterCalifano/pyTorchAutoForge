# sub_package/__init__.py
import importlib

def lazy_import(module_name):
    module = None

    def load():
        nonlocal module
        if module is None:
            module = importlib.import_module(
                f'.{module_name}', package=__name__)
        return module
    return load


# Load all modules in all sub-packages
sub_dirs = ['onnx', 'tcp', 'torch']

for subpackage in sub_dirs:
    globals()[subpackage] = lazy_import(subpackage)
