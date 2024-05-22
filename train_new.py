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
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger


from datasets.dataset_factory import get_module as get_dataset
from experiments.experiment_factory import get_module as get_experiment
from experiments.MNIST_salome_mil_experiment import MILEXperiment_CV
from models.model_factory import get_module

terminal_logger = logging.getLogger(__name__)

def set_training_env_settings(consistent = True):
    cudnn.deterministic = consistent
    cudnn.benchmark = not consistent
    os.environ["HYDRA_FULL_ERROR"] = "1"

def initialize_logger(config,split_num = -1):
    wnb_logger = WandbLogger(log_model=False,save_dir=config["logging_params"]["save_dir"]+f"split{split_num}/",name = config["logging_params"]["name"],project=config["logging_params"]["project_name"])
    wnb_logger.log_hyperparams(params=config)#this addes all my hyperparameters to be tracked by wandb gs
    return wnb_logger

def get_and_configure_callbacks(config,logger):
    checkpoint_callback = ModelCheckpoint(
            dirpath=f"{logger.save_dir}",
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
            filename="{epoch}-{val_loss:.4f}",
            verbose=True,
            save_last=True,
            every_n_epochs=3,
        )
    early_stopping = EarlyStopping(
        monitor=config["ES_params"]["monitor"],
        min_delta=config["ES_params"]["min_delta"],
        patience=config["ES_params"]["patience"],
        )
    return checkpoint_callback, early_stopping

@hydra.main(config_path="configs", config_name="RBC_topoMIL")
def main(cfg: DictConfig) -> None:
    print(cfg)
    config = OmegaConf.to_container(cfg)
    seed = config["logging_params"]["manual_seed"]

    set_training_env_settings(True)

    data = get_dataset(config["dataset"]["name"], config["dataset"]["config"])

    k_folds = config["dataset"]["config"]["k_fold"]
    results = []
    for split_index in range(abs(k_folds)):
        
        wnb_logger  = initialize_logger(config,split_index)

        checkpoint_callback, early_stopping = get_and_configure_callbacks(config,wnb_logger)
        
        model = get_module(config["model_params"]["name"], config["model_params"]["config"])
        experiment = get_experiment(config["exp_type"],config["exp_params"],model)   
        terminal_logger.info(f"The logging path is {wnb_logger.save_dir}") 

        runner = Trainer(
            enable_checkpointing=True,
            logger=wnb_logger,
            #callbacks=[early_stopping],
            **config["trainer_params"],
            
        )

        terminal_logger.info(f"======= Training {config['model_params']['name']} =======")
        
        runner.fit(experiment, data)
        
        result =runner.test(experiment, data, ckpt_path="best")[0]
        results.append(result)


        #debug log smthn
        #terminal_logger.info(f"{wnb_logger.save_dir}") 
    global output_results
    output_results= results #returns the results of every split permutation

def set_config_file_environment_variable(folder_path,file_name):
        sys.argv[2] = folder_path
        sys.argv[4] = file_name

def initialize_config_env():
    sys.argv.append("--config-path")
    sys.argv.append("123")
    sys.argv.append("--config-name")
    sys.argv.append("123")
    sys.argv.append("hydra.run.dir=./")

def setup_and_start_training(number_of_runs = 1):
    configs = [
        #["configs/test/","topo_test_cubical.yaml"],
        #["configs/test/","test.yaml"],
        ["configs/test/","SCEMILA_test.yaml"],
        ]
    

    initialize_config_env()
    for test_index in range(len(configs)):
        set_config_file_environment_variable(configs[test_index][0],configs[test_index][1])
        all_metrics = []
        print(f"Running the configuration {configs[test_index][0]}{configs[test_index][1]}")
        for run in range(number_of_runs):
            print(f"Starting run #{run+1}")
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
        
 

        table = tabulate(results, headers=["Metric", "Mean", "Std"], tablefmt="grid")

        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%H.%M.%d.%m")
        file_name = (
            f"{configs[test_index][0]}{configs[test_index][1]}_{formatted_datetime}.txt"
        )
        with open(file_name, "w") as file:
            file.write(table)

     
if __name__ == "__main__":
    setup_and_start_training(1)
    



