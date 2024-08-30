import torch
import datetime
import os
import torch
import logging
import sys
import hydra
import torch.backends.cudnn as cudnn
from tabulate import tabulate

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb


from datasets.dataset_factory import get_module as get_dataset
from experiments.experiment_factory import get_module as get_experiment
from models.model_factory import get_module



def set_training_env_settings(consistent = False):
    cudnn.deterministic = not consistent
    cudnn.benchmark = consistent
    os.environ["HYDRA_FULL_ERROR"] = "1"
    torch.set_float32_matmul_precision('medium')
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def initialize_logger(config,split_num = -1):
    save_dir = os.path.join(config["logging_params"]["save_dir"], f"split{split_num}/")
    os.makedirs(save_dir, exist_ok=True)
    wnb_logger = WandbLogger(log_model=False,save_dir=config["logging_params"]["save_dir"]+f"split{split_num}/",name = config["logging_params"]["name"],project=config["logging_params"]["project_name"],config = config)
    wnb_logger.log_hyperparams(params=config)#this addes all my hyperparameters to be tracked by wandb gs
    version_number = wnb_logger.version
    wandb.run.summary["version_number_sum"] = version_number
    return wnb_logger

def get_and_configure_callbacks(config,logger):    
    checkpoint_callback = ModelCheckpoint(
            #dirpath=f"{logger.save_dir}",
            monitor="val_mil_loss_epoch",
            mode="min",
            save_weights_only=True,
            filename="{epoch}-{val_mil_loss_epoch:.4f}",
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

@hydra.main(config_path="configs/SCEMILA_approaches/topo/", config_name="topo_dino_image_input.yaml",version_base=None)
def main(cfg: DictConfig) -> None:
    
    
    config = OmegaConf.to_container(cfg)
    print(config)
    seed = config["logging_params"]["manual_seed"]

    set_training_env_settings(False)
    print(f"cudnn.deterministic: {cudnn.deterministic}")

    data = get_dataset(config["dataset"]["name"], config["dataset"]["config"])

    k_folds = config["dataset"]["config"]["k_fold"]
    results = []
    for split_index in range(abs(k_folds)):
        wnb_logger  = initialize_logger(config,split_index)
        print(config)
        checkpoint_callback, early_stopping = get_and_configure_callbacks(config,wnb_logger)
        
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
        
        result =runner.test(experiment, data, ckpt_path="best")[0]
        results.append(result)
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
            #["configs/SCEMILA_approaches/normal/","dino_input.yaml"],
            #["configs/SCEMILA_approaches/normal/","fnl34_input.yaml"],
            # ["configs/SCEMILA_approaches/normal/","gray_image_input.yaml"],
            ["configs/SCEMILA_approaches/normal/","image_input.yaml"],
            #### topo methods
            #["configs/SCEMILA_approaches/topo/","topo_gray_image_image_input.yaml"],
            #["configs/SCEMILA_approaches/topo/","topo_image_image_input.yaml"],
            #["configs/SCEMILA_approaches/topo/","topo_dino_image_input.yaml"],
            #["configs/SCEMILA_approaches/topo/","topo_dino_dino_input.yaml"]
            ]
    

    initialize_config_env()
    for test_index in range(len(configs)):
        set_config_file_environment_variable(configs[test_index][0],configs[test_index][1])
        all_metrics = []
        print(f"Running the configuration {configs[test_index][0]}{configs[test_index][1]}")
        for run in range(number_of_runs):
            print(f"Starting {run+1}th run")
            main()
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
    setup_and_start_training(5)