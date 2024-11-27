import importlib
import typing

MODULES = {
    "salome_experiment_MIL_CV": "experiments.MNIST_salome_mil_experiment.MILEXperiment_CV",
    "salome_experiment_topo_MIL_CV": "experiments.MNIST_salome_mil_experiment.TopoRegMILEXperiment_CV",
    "SCEMILA_experiment": "experiments.SCEMILA_experiment.SCEMILA_Experiment",
    "topo_SCEMILA_experiment": "experiments.SCEMILA_topo_experiment.TopoSCEMILA_Experiment",
}

NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def lazy_import(path: str):
    """Lazy imports a module and retrieves the required class or function."""
    module_path, class_name = path.rsplit(".", 1)  # Split into module and class name
    module = importlib.import_module(module_path)  # Dynamically import the module
    return getattr(module, class_name)  # Retrieve the class or function


def get_module(name: str, config: typing.Dict[str, typing.Any], model, data=None):
    """Recursively deserializes objects registered in MODULES."""
    if name not in MODULES:
        raise KeyError(
            f"{name} not found in registered modules. Available are {list(MODULES.keys())}."
        )
    cls_path = MODULES[name]
    cls = lazy_import(cls_path)  # Dynamically load the class
    exp = cls(model, config)  # Instantiate the experiment class
    if data is not None and hasattr(exp, "set_dataset_for_latent_visualization"):
        exp.set_dataset_for_latent_visualization(data)  # Set the dataset if the method exists
    return exp
