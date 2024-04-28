from configs.sweep_configs.test_sweep import TEST_DICT
from train import setup_and_start_training,main
import wandb


def sweep_parameters(sweep_config =TEST_DICT,project_name="my-first-sweep"):
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)  
    wandb.agent(sweep_id, function=main, count=10)  