import os
import glob
import importlib

# Get the current directory path
current_dir = os.path.dirname(__file__)

# Find all .py files in the current directory, excluding __init__.py
modules = glob.glob(os.path.join(current_dir, "*.py"))
modules = [os.path.basename(f)[:-3] for f in modules if os.path.isfile(f) and f.endswith(".py") and f != "__init__.py"]

__all__ = []

# Import all modules dynamically and add classes to __all__
for module in modules:
    try:
        mod = importlib.import_module(f"datasets.SCEMILA.{module}")
        for attribute_name in dir(mod):
            attribute = getattr(mod, attribute_name)
            if isinstance(attribute, type):
                globals()[attribute_name] = attribute
                __all__.append(attribute_name)
    except Exception as e:
        print(f"Failed to import module: {module}, error: {e}")
