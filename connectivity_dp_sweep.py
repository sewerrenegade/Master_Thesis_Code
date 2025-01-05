from models.topology_models.custom_topo_tools.connectivity_dp_hyperparameter_sweeper import ConnectivityDPHyperparameterSweeper
from models.topology_models.custom_topo_tools.connectivity_dp_sweep_configs import ALL_PERUMTAIONS, LR_OPTIMIZER_SWEEP, IMPORTANCE_STRAT_SWEEP
if __name__ == "__main__":
    # x = ConnectivityHyperParamExperiment()
    # output = x.run_experiment()
    # print(output)
    config = LR_OPTIMIZER_SWEEP
    sweeper = ConnectivityDPHyperparameterSweeper(config, "lr_optimizier_Dinobloom_test")
    sweeper.sweep()

# enc, lab = get_labeled_dinobloom_encodings()
# np.savez("data/SCEMILA/dinbloomS_labeled.npz",embedding  = enc,labels = lab)