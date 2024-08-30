import scipy
import torch
from scipy import io
import numpy as np
from sklearn.model_selection import KFold
import torch.optim as optim
import wandb



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


from datasets.dataset_factory import get_module as get_dataset
from experiments.experiment_factory import get_module as get_experiment
from experiments.MNIST_salome_mil_experiment import MILEXperiment_CV
from models.model_factory import get_module

logger = logging.getLogger(__name__)

test_result = None

@hydra.main(config_path="configs", config_name="RBC_topoMIL")
def main(cfg: DictConfig) -> None:
    print(cfg)
    config = OmegaConf.to_container(cfg)
    cfg =config
    seed = cfg["logging_params"]["manual_seed"]
    cudnn.deterministic = True
    cudnn.benchmark = False
    data = get_dataset(config["dataset"]["name"], config["dataset"]["config"])
    for split_index in range(1):
        # tb_logger = TensorBoardLogger(
        #     save_dir=cfg["logging_params"]["save_dir"]+f"split{split_index}/",
        #     name=cfg["logging_params"]["name"],
        # )
        # tb_logger.log_hyperparams(params=cfg)
        wnb_logger = WandbLogger(log_model=False,save_dir=cfg["logging_params"]["save_dir"]+f"split{split_index}/",name = cfg["logging_params"]["name"],project='my-first-sweep')
        wnb_logger.log_hyperparams(params=cfg)#this addes all my hyperparameters to be tracked by wandb
        early_stopping = EarlyStopping(
        monitor=cfg["ES_params"]["monitor"],
        min_delta=cfg["ES_params"]["min_delta"],
        patience=cfg["ES_params"]["patience"],
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{wnb_logger.save_dir}",
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
            filename="{epoch}-{val_loss:.2f}",
            verbose=True,
            save_last=True,
            every_n_epochs=3,
        )
        model = get_module(cfg["model_params"]["name"], cfg["model_params"]["config"])
        experiment = get_experiment(cfg["exp_type"],cfg["exp_params"],model)   
        logger.info(f"{wnb_logger.save_dir}") 
        runner = Trainer(
            enable_checkpointing=True,
            logger=wnb_logger,
            #callbacks=[early_stopping],
            **cfg["trainer_params"],
            
        )

        logger.info(f"======= Training {cfg['model_params']['name']} =======")
        runner.fit(experiment, data)
        global test_result
        try:
            test_result.append(runner.test(experiment, data, ckpt_path="best")[0])
        except Exception as e:
            print(e)
        #global best_accuracies
        #best_accuracies.append(experiment.best_acc)

        print("run is done!")
        
       # print(f"Run #{run}, split {split_index} completed with acc {test_result[-1]['Accuracy']} ")


def set_config_file_environment_variable(folder_path,file_name):
        sys.argv[2] = folder_path
        sys.argv[4] = file_name

def initialize_config_env():
    sys.argv.append("--config-path")
    sys.argv.append("123")
    sys.argv.append("--config-name")
    sys.argv.append("123")
    sys.argv.append("hydra.run.dir=./")

def setup_and_start_training():
    configs = [
        #["configs/test/","topo_test_reconstruction.yaml"],
        
        #["configs/test/","test.yaml"],
        ["configs/test/","topo_test_cubical.yaml"],
        ]
    #os.environ["HYDRA_FULL_ERROR"] = "1"
    initialize_config_env()



    for test_index in range(len(configs)):
        set_config_file_environment_variable(configs[test_index][0],configs[test_index][1])
        all_metrics = []
        print(sys.argv)
        for run in range(1):
            print(f"Starting run {run+1}")
            test_result=[]
            main()
            all_metrics.extend(test_result)

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
    setup_and_start_training()
    



