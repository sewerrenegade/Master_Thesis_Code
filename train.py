import scipy
import torch
from scipy import io
import numpy as np
from sklearn.model_selection import KFold
import torch.optim as optim
import copy



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
    for split_index in range(2):
        # tb_logger = TensorBoardLogger(
        #     save_dir=cfg["logging_params"]["save_dir"]+f"split{split_index}/",
        #     name=cfg["logging_params"]["name"],
        # )
        # tb_logger.log_hyperparams(params=cfg)
        wnb_logger = WandbLogger(log_model="all",save_dir=cfg["logging_params"]["save_dir"]+f"split{split_index}/",name = "testing")
        wnb_logger.log_hyperparams(params=cfg)


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
        # logger.info(f"{wb_logger.save_dir}")

            
        runner = Trainer(
            enable_checkpointing=True,
            logger=wnb_logger,
            #callbacks=[early_stopping],
            **cfg["trainer_params"],
            
        )



        
        logger.info(f"======= Training {cfg['model_params']['name']} =======")
        runner.fit(experiment, data)
        global test_result
        test_result.append(runner.test(experiment, data, ckpt_path="best")[0])
        global best_accuracies
        #best_accuracies.append(experiment.best_acc)

        print("run is done!")
        
        print(f"Run #{run}, split {split_index} completed with acc {test_result[-1]['Accuracy']} best acc {experiment.best_acc}")


def set_config_file_environment_variable(folder_path,file_name):
        sys.argv[2] = folder_path
        sys.argv[4] = file_name

def initialize_config_env():
    sys.argv.append("--config-path")
    sys.argv.append("123")
    sys.argv.append("--config-name")
    sys.argv.append("123")
    sys.argv.append("hydra.run.dir=./")

     
if __name__ == "__main__":
    configs = [
        ["configs/test/","topo_test.yaml"],
        ["configs/test/","test.yaml"],

        # #Vanilla RGMIL
        

        ]
    #os.environ["HYDRA_FULL_ERROR"] = "1"
    initialize_config_env()



    for test_index in range(len(configs)):
        set_config_file_environment_variable(configs[test_index][0],configs[test_index][1])
        all_metrics = []
        all_best_accuracies = [] 
        print(sys.argv)
        for run in range(1):
            print(f"Starting run {run+1}")
            test_result=[]
            best_accuracies = []
            main()
            all_metrics.extend(test_result)
            all_best_accuracies.extend(best_accuracies)

        mean_metrics = {
            metric: torch.mean(torch.tensor([m[metric] for m in all_metrics]))
            for metric in all_metrics[0]
        }
        std_metrics = {
            metric: torch.std(torch.tensor([m[metric] for m in all_metrics]))
            for metric in all_metrics[0]
        }
        results = []
        all_best_accuracies = np.array(all_best_accuracies)
        mean_best_acc = np.mean(all_best_accuracies)
        std_dev_best_acc = np.std(all_best_accuracies)
        results.append(
                [
                    "Best Acc",
                    f"{mean_best_acc:.4f}",
                    f"+/- {std_dev_best_acc:.4f}",
                ]
            )
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
            f"{configs[test_index][1]}{configs[test_index][0]}_{formatted_datetime}.txt"
        )
        with open(file_name, "w") as file:
            file.write(table)




