import importlib

MODULES = {
    "CV_MIL": "models.salome_models.mil.CV_MIL",
    "CV_TopoRegMIL": "models.salome_models.topo_reg_mil.CV_TopoRegMIL",
    "Adam": "models.salome_models.optimizer.Adam",
    "ConstantScheduler": "models.salome_models.scheduler.ConstantScheduler",
    "SCEMILA": "models.SCEMILA.SCEMILA_model.AMiL",
    "TopoSCEMILA": "models.SCEMILA.topo_SCEMILA_model.TopoAMiL",
    "SGD": "torch.optim.SGD",
}

NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def lazy_import(path: str):
    """Lazy imports a module and retrieves the required class or function."""
    module_path, class_name = path.rsplit(".", 1)  # Split into module and class name
    module = importlib.import_module(module_path)  # Dynamically import the module
    return getattr(module, class_name)  # Retrieve the class or function


def get_module(name: str, config):
    """Recursively deserializes objects registered in MODULES."""
    if name not in MODULES:
        raise KeyError(
            f"{name} not found in registered modules. Available are {list(MODULES.keys())}."
        )
    cls_path = MODULES[name]
    cls = lazy_import(cls_path)  # Dynamically load the class
    for key, value in config.items():
        if isinstance(value, dict) and NAME_KEY in value and CONFIG_KEY in value:
            config[key] = get_module(value[NAME_KEY], value[CONFIG_KEY])
    return cls(**config)
