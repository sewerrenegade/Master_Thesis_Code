print("Started Execution, importing...")
import argparse
import torch
import datetime
import os
import torch
import logging
import sys
import hydra
import torch.backends.cudnn as cudnn
from tabulate import tabulate
import re

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb






def set_training_env_settings(consistent = False):
    print(f"cudnn.enabled: {cudnn.enabled}")
    cudnn.deterministic = False #was true
    cudnn.benchmark = False #was false3
    os.environ["HYDRA_FULL_ERROR"] = "1"
    torch.set_float32_matmul_precision('medium')
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def initialize_logger(config,split_num = -1,k_folds = -1):
    save_dir = os.path.join(config["logging_params"]["save_dir"])
    
    os.makedirs(save_dir, exist_ok=True)
    if k_folds > 1:
        run_name= config["logging_params"]["name"]+f"_split_{split_num}_of_{k_folds}"
    else:
        run_name= config["logging_params"]["name"]
        
    wnb_logger = WandbLogger(log_model=False,
                            save_dir=config["logging_params"]["save_dir"],
                            name = run_name,
                            project=config["logging_params"]["project_name"],
                            # config = config,
                            entity = "milad-research")
    #wnb_logger
    wnb_logger.log_hyperparams(params=config)#this addes all my hyperparameters to be tracked by wandb gs, turns out the lighting constructor of wandb is doing the job, this is redundant
    wandb.run.summary["version_number_sum"] = wnb_logger.version
    return wnb_logger

def get_and_configure_callbacks(config):
    checkpoint_callback = ModelCheckpoint(
            monitor="val_correct_epoch",
            mode="max",
            save_weights_only=True,
            filename="{epoch}-{val_correct_epoch:.4f}",
            verbose=True,
            save_last=True,
            every_n_epochs=1,
            save_top_k = 3
        )
    early_stopping = EarlyStopping(
        monitor=config["ES_params"]["monitor"],
        min_delta=config["ES_params"]["min_delta"],
        patience=config["ES_params"]["patience"],
        mode = config["ES_params"]["mode"]
        )
    return checkpoint_callback, early_stopping

def get_hydra_override_args():
    print(sys.argv)
    overrides=[arg.lstrip('--') for arg in sys.argv][1:]
    print(overrides)
    hydra_overides = []
    for item in overrides:
        if item.startswith('base_config_path='):
            path = item.split('=', 1)[1]
            print(f"base_config_path {path}")
            continue

        if item.startswith('config_folder='):
            path = item.split('=', 1)[1]
            print(f"config_folder {path}")
            continue

        if item.startswith('config_name='):
            path = item.split('=', 1)[1]
            print(f"config_name {path}")
            continue
        hydra_overides.append(item)
    print(f"Hydra overrides are {hydra_overides}")
    return hydra_overides
    
    
    #
#@hydra.main(config_path="configs/SCEMILA_approaches/normal/", config_name="opt_image_input.yaml",version_base=None)
def main(config_path="configs/SCEMILA_approaches/normal/", config_name="opt_image_input.yaml",version_base=None) -> None:
    with hydra.initialize(config_path=config_path,version_base=None):
        cfg = hydra.compose(config_name=config_name,overrides=get_hydra_override_args())
    config = OmegaConf.to_container(cfg)
    print(f"This is the config file \n  {config}")
    seed = config["logging_params"]["manual_seed"]

    set_training_env_settings(False)
    print(f"cudnn.deterministic: {cudnn.deterministic}")
    from datasets.dataset_factory import get_module as get_dataset
    from experiments.experiment_factory import get_module as get_experiment
    from models.model_factory import get_module

    data = get_dataset(config["dataset"]["name"], config["dataset"]["config"])

    k_folds = config["dataset"]["config"]["k_fold"]
    results = []
    for split_index in range(abs(k_folds)):
        wnb_logger  = initialize_logger(config,split_index,k_folds)
        print(config)
        checkpoint_callback, early_stopping = get_and_configure_callbacks(config)
        
        model = get_module(config["model_params"]["name"], config["model_params"]["config"])
        experiment = get_experiment(config["exp_type"],config["exp_params"],model)
        terminal_logger = logging.getLogger(__name__)
        terminal_logger.info(f"The logging path is {wnb_logger.save_dir}") 

        runner = Trainer(
            logger=wnb_logger,
            callbacks=[early_stopping,checkpoint_callback],
            log_every_n_steps=1,
            **config["trainer_params"]
        )

        terminal_logger.info(f"======= Training {config['model_params']['name']} =======")
        
        runner.fit(experiment, data)
        
        result =runner.test(experiment, data, ckpt_path="best")
        best_ckpt_path = checkpoint_callback.best_model_path
        wandb.run.summary["test_checkpoint_path"] = best_ckpt_path
        match = re.search(r'epoch=(\d+)', best_ckpt_path)
        if match:
            epoch_number = int(match.group(1))
            wandb.run.summary["best_checkpoint_epoch"] = epoch_number
            #print(f"Extracted epoch number: {epoch_number}")
        else:
            print("No epoch number found in the string.")
        results.append(result[0])
        wandb.finish()

        #debug log smthn
        #terminal_logger.info(f"{wnb_logger.save_dir}") 
    global output_results
    output_results= results #returns the results of every split permutation

def set_config_file_environment_variable(folder_path,file_name):
    sys.argv[2] = folder_path
    sys.argv[4] = file_name
    
def save_important_results_next_to_config(results,config):
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%H.%M.%d.%m")
        file_name = (
            f"{config[0]}{config[1]}_{formatted_datetime}.txt"
        )
        table = tabulate(results, headers=["Metric", "Mean", "Std"], tablefmt="grid")
        with open(file_name, "w") as file:
            file.write(table)
    

def initialize_config_env():
    sys.argv.append("--config-path")
    sys.argv.append("123")
    sys.argv.append("--config-name")
    sys.argv.append("123")
    sys.argv.append("hydra.run.dir=./")

def setup_and_start_training(number_of_runs = 1):
    configs = [
            #["configs/SCEMILA_approaches/normal/","test_image_input.yaml"],
            #["configs/SCEMILA_approaches/normal/","opt_image_input.yaml"],
            #["configs/SCEMILA_approaches/normal/","opt_kfold_image_input.yaml"],
            #["configs/SCEMILA_approaches/normal/","dino_input.yaml"],
            #["configs/SCEMILA_approaches/normal/","fnl34_input.yaml"],
            # ["configs/SCEMILA_approaches/normal/","gray_image_input.yaml"],
            #["configs/SCEMILA_approaches/normal/","image_input.yaml"],
            #### topo methods
            #["configs/SCEMILA_approaches/topo/","topo_gray_image_image_input.yaml"],
            #["configs/SCEMILA_approaches/topo/","topo_image_image_input.yaml"],
            ["configs/SCEMILA_approaches/topo/","topo_dino_image_input.yaml"],
            #["configs/SCEMILA_approaches/topo/","topo_dino_dino_input.yaml"]
            ]
    
    for test_index in range(len(configs)):
        #set_config_file_environment_variable(configs[test_index][0],configs[test_index][1])
        all_metrics = []
        print(f"Running the configuration {configs[test_index][0]}{configs[test_index][1]}")
        for run in range(number_of_runs):
            print(f"Starting {run+1}th run")
            main(config_path=configs[test_index][0],config_name=configs[test_index][1])
            all_metrics.extend(output_results)

        mean_metrics = {
            metric: torch.mean(torch.tensor([m[metric] for m in all_metrics]))
            for metric in all_metrics[0]
        }
        std_metrics = {
            metric: torch.std(torch.tensor([m[metric] for m in all_metrics]))
            for metric in all_metrics[0]
        }
        results = []
        for metric in mean_metrics:
            results.append(
                [
                    metric,
                    f"{mean_metrics[metric]:.4f}",
                    f"+/- {std_metrics[metric]:.4f}",
                ]
            )
        save_important_results_next_to_config(results,configs[test_index])

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training with specified configuration.")
    parser.add_argument('--config_folder', type=str,default="configs/SCEMILA_approaches/topo/" ,help="Path to the configuration folder.")
    parser.add_argument('--config_name', type=str,default="no_gpu_topo_eucl_image_input.yaml", help="Name of the configuration file.")

    args = parser.parse_args()
    main(config_path=args.config_folder,config_name=args.config_name)

