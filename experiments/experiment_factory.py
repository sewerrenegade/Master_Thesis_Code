import typing
from experiments.MNIST_salome_mil_experiment import MILEXperiment_CV,TopoRegMILEXperiment_CV
from experiments.SCEMILA_experiment import SCEMILA_Experiment
from experiments.SCEMILA_topo_experiment import TopoSCEMILA_Experiment


MODULES = {
    "salome_experiment_MIL_CV": MILEXperiment_CV,
    "salome_experiment_topo_MIL_CV": TopoRegMILEXperiment_CV,
    "SCEMILA_experiment": SCEMILA_Experiment,
    "topo_SCEMILA_experiment": TopoSCEMILA_Experiment
}

NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def get_module(name: str, config: typing.Dict[str, typing.Any],model,data=None):
    """Recursively deserializes objects registered in MODULES."""
    if name not in MODULES:
        raise KeyError(
            f"{name} not found in registered modules. Available are {MODULES.keys()}."
        )
    cls = MODULES[name]
    exp= cls(model,config)
    if data is not None and hasattr(exp,"set_dataset_for_latent_visualization"):
        exp.set_dataset_for_latent_visualization(data)
    return exp