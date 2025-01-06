from models.topology_models.custom_topo_tools.connectivity_dp_hyperparameter_sweeper import ConnectivityDPHyperparameterSweeper
from models.topology_models.custom_topo_tools.connectivity_dp_sweep_configs import (ALL_PERUMTAIONS, LR_OPTIMIZER_SWEEP,
                                                                                    IMPORTANCE_STRAT_SWEEP,NORMALIZE_INPUT_SWEEP,
                                                                                    WEIGHT_DECAY_SWEEP, AUGMENTATION_STRENGTH_SWEEP)

SWEEP_CONFIGS ={
    "lr_optimizer_sweep": LR_OPTIMIZER_SWEEP,
    "importance_strat_sweep": IMPORTANCE_STRAT_SWEEP,
    "normalize_input_sweep": NORMALIZE_INPUT_SWEEP,
    "weight_decay_sweep": WEIGHT_DECAY_SWEEP,
    "augmentation_strength_sweep": AUGMENTATION_STRENGTH_SWEEP
}
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run sweep with specified configuration.")
    parser.add_argument('--config_name', type=str,default="config name not passed", help="Name of the configuration file.")
    args = parser.parse_args()
    config_name = args.config_name
    if config_name in SWEEP_CONFIGS:
        sweeper = ConnectivityDPHyperparameterSweeper(SWEEP_CONFIGS[config_name], config_name)
        sweeper.sweep()
    else:
        print(f"The configuration {config_name} is not a valid configuration name. Valid configs are {list(SWEEP_CONFIGS.keys())}")
