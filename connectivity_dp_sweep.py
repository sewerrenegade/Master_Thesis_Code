from models.topology_models.prop_topo_tools.connectivity_dp_hyperparameter_sweeper import ConnectivityDPHyperparameterSweeper
from models.topology_models.prop_topo_tools.connectivity_dp_sweep_configs import (ALL_PERUMTAIONS, LR_OPTIMIZER_SWEEP,
                                                                                    IMPORTANCE_STRAT_SWEEP,NORMALIZE_INPUT_SWEEP,
                                                                                    WEIGHT_DECAY_SWEEP, AUGMENTATION_STRENGTH_SWEEP,OPT_PERMUTATIONS_CLUSTERS,
                                                                                    OPT_PERMUTATIONS_DINO,OPT_PERMUTATIONS_MNSIT,
                                                                                    OPT_PERMUTATIONS_SWISS,DATA_SIZE_SWEEP,LR_OPTIMIZER_SWEEP_2)

SWEEP_CONFIGS ={
    "lr_optimizer_sweep": LR_OPTIMIZER_SWEEP,
    "lr_optimizer_sweep_2": LR_OPTIMIZER_SWEEP_2,
    "importance_strat_sweep": IMPORTANCE_STRAT_SWEEP,
    "data_size_sweep": DATA_SIZE_SWEEP,
    "normalize_input_sweep": NORMALIZE_INPUT_SWEEP,
    "weight_decay_sweep": WEIGHT_DECAY_SWEEP,
    "augmentation_strength_sweep": AUGMENTATION_STRENGTH_SWEEP,
    "opt_MNIST": OPT_PERMUTATIONS_MNSIT,
    "opt_swiss": OPT_PERMUTATIONS_SWISS,
    "opt_dino": OPT_PERMUTATIONS_DINO,
    "opt_clusters": OPT_PERMUTATIONS_CLUSTERS
}
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run sweep with specified configuration.")
    parser.add_argument('--config_name', type=str,default="config name not passed", help="Name of the configuration file.")
    parser.add_argument('--nb_repeats', type=int,default=1, help="Name of the configuration file.")
    args = parser.parse_args()
    config_name = args.config_name
    nb_repeats = int(args.nb_repeats)
    if config_name in SWEEP_CONFIGS:
        sweeper = ConnectivityDPHyperparameterSweeper(SWEEP_CONFIGS[config_name], config_name,n_repeats=nb_repeats)
        sweeper.sweep()
    else:
        print(f"The configuration {config_name} is not a valid configuration name. Valid configs are {list(SWEEP_CONFIGS.keys())}")


# sbatch run_connectivity_sweep_on_hpc.sh lr_optimizer_sweep
# sbatch run_connectivity_sweep_on_hpc.sh importance_strat_sweep
# sbatch run_connectivity_sweep_on_hpc.sh data_size_sweep
# sbatch run_connectivity_sweep_on_hpc.sh normalize_input_sweep
# sbatch run_connectivity_sweep_on_hpc.sh weight_decay_sweep
# sbatch run_connectivity_sweep_on_hpc.sh augmentation_strength_sweep
# sbatch run_connectivity_sweep_on_hpc.sh opt_MNIST
# sbatch run_connectivity_sweep_on_hpc.sh opt_swiss
# sbatch run_connectivity_sweep_on_hpc.sh opt_dino
# sbatch run_connectivity_sweep_on_hpc.sh opt_clusters
