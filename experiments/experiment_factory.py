import typing

from experiments.MNIST_salome_mil_experiment import MILEXperiment_CV,TopoRegMILEXperiment_CV
from experiments.SCEMILA_experiment import SCEMILA_Experiment


MODULES = {
    "salome_experiment_MIL_CV": MILEXperiment_CV,
    "salome_experiment_topo_MIL_CV": TopoRegMILEXperiment_CV,
    "SCEMILA_experiment": SCEMILA_Experiment
}

NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def get_module(name: str, config: typing.Dict[str, typing.Any],model):
    """Recursively deserializes objects registered in MODULES."""
    if name not in MODULES:
        raise KeyError(
            f"{name} not found in registered modules. Available are {MODULES.keys()}."
        )
    cls = MODULES[name]
    return cls(model,config)