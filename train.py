import numpy
import os



def set_training_env_settings():
    seed = 42
    from numpy.random import seed as np_seed
    from random import seed as rng_seed
    from torch import set_float32_matmul_precision,manual_seed
    manual_seed(seed)
    np_seed(seed)
    rng_seed(seed)
    import torch.backends.cudnn as cudnn
    print(f"cudnn.enabled: {cudnn.enabled}")
    cudnn.deterministic = True #was true
    cudnn.benchmark = False #was false3
    os.environ["HYDRA_FULL_ERROR"] = "1"
    set_float32_matmul_precision('medium')

    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def initialize_logger(config,split_num = -1,k_folds = -1):
    save_dir = os.path.join(config["logging_params"]["save_dir"])
    from pytorch_lightning.loggers import WandbLogger
    os.makedirs(save_dir, exist_ok=True)
    if k_folds > 1:
        run_name= config["logging_params"]["name"]+f"_split_{split_num + 1}_of_{k_folds}"
    else:
        run_name= config["logging_params"]["name"]
        
    wnb_logger = WandbLogger(log_model=False,
                            save_dir=config["logging_params"]["save_dir"],
                            name = run_name,
                            project=config["logging_params"]["project_name"],
                            config = config,
                            entity = "milad-research")
    wnb_logger.experiment.summary["version_number_sum"] = "i <3 light"
    wnb_logger.experiment.summary["version_number_sum"] = wnb_logger.version
    if k_folds > 1:
        wnb_logger.experiment.summary["split_index"] = split_num + 1
    return wnb_logger

def get_and_configure_callbacks(config):
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from trainer_scripts.custom_model_checkpoint import CustomModelCheckpoint
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
    checkpoint_callback = CustomModelCheckpoint(
        save_weights_only=True,
        filename="{epoch}-{val_correct_epoch:.4f}-{val_loss_epoch:.4f}",
        verbose=True,
        save_last=True,
        top_k_to_save=3
    )
    early_stopping = EarlyStopping(
        monitor=config["ES_params"]["monitor"],
        min_delta=config["ES_params"]["min_delta"],
        patience=config["ES_params"]["patience"],
        mode = config["ES_params"]["mode"]
        )
    return checkpoint_callback, early_stopping

def get_hydra_override_args():    
    from sys import argv
    print(argv)
    overrides=[arg.lstrip('--') for arg in argv][1:]
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
    import numpy
    from hydra import initialize, compose
    from omegaconf import OmegaConf

    with initialize(config_path=config_path,version_base=None):
        cfg = compose(config_name=config_name,overrides=get_hydra_override_args())
    config = OmegaConf.to_container(cfg)
    print(f"This is the config file \n  {config}")
    seed = config["logging_params"]["manual_seed"]

    set_training_env_settings()
    from datasets.dataset_factory import get_module as get_dataset
    from experiments.experiment_factory import get_module as get_experiment
    from models.model_factory import get_module
    from pytorch_lightning import Trainer
    from logging import getLogger
    data = get_dataset(config["dataset"]["name"], config["dataset"]["config"])

    k_folds = config["dataset"]["config"]["k_fold"]
    results = []
    for split_index in range(abs(k_folds)):
        wnb_logger  = initialize_logger(config,split_index,k_folds)
        print(wnb_logger.version)
        checkpoint_callback, early_stopping = get_and_configure_callbacks(config)
        
        model = get_module(config["model_params"]["name"], config["model_params"]["config"])
        experiment = get_experiment(config["exp_type"],config["exp_params"],model)
        terminal_logger = getLogger(__name__)
        terminal_logger.info(f"The logging path is {wnb_logger.save_dir}") 

        runner = Trainer(
            logger=wnb_logger,
            callbacks=[early_stopping,checkpoint_callback],
            log_every_n_steps=1,
            **config["trainer_params"]
        )


        terminal_logger.info(f"======= Training {config['model_params']['name']} =======")
        
        runner.fit(experiment, data)
        best_ckpt_path = checkpoint_callback.get_best_path()
        result =runner.test(experiment, data, ckpt_path=best_ckpt_path)
        
        wnb_logger.experiment.summary["test_checkpoint_path"] = best_ckpt_path

        from re import search
        match = search(r'epoch=(\d+)', best_ckpt_path)
        if match:
            epoch_number = int(match.group(1))
            wnb_logger.experiment.summary["best_checkpoint_epoch"] = epoch_number
            #print(f"Extracted epoch number: {epoch_number}")
        else:
            print("No epoch number found in the string.")
        from results.model_visualisation.instance_bag_SCEMILA_visulaizer import get_bag_and_instance_level_2D_embeddings
        get_bag_and_instance_level_2D_embeddings(model= model,dataset=data)
        results.append(result[0])
        wnb_logger.experiment.finish()
        #debug log smthn
        #terminal_logger.info(f"{wnb_logger.save_dir}") 
    global output_results
    output_results= results #returns the results of every split permutation





if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run training with specified configuration.")
    parser.add_argument('--config_folder', type=str,default="configs/SCEMILA_approaches/topo/" ,help="Path to the configuration folder.")
    parser.add_argument('--config_name', type=str,default="no_gpu_topo_eucl_image_input.yaml", help="Name of the configuration file.")
    args = parser.parse_args()

    main(config_path=args.config_folder,config_name=args.config_name)

