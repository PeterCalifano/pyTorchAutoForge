import pkgutil
import importlib
import pyTorchAutoForge

def test_import_pytorch_autoforge_modules():
    package_name = "pyTorchAutoForge"
    errors = []

    # Iterate through all submodules in pyTorchAutoForge
    for _, mod_name, _ in pkgutil.walk_packages(pyTorchAutoForge.__path__, package_name + "."):
        try:
            importlib.import_module(mod_name)
        except ImportError as e:
            errors.append(f"Failed to import {mod_name}: {e}")

    # If there are any errors, pytest will fail the test
    assert not errors, f"Some modules failed to import:\n" + "\n".join(errors)


    




 