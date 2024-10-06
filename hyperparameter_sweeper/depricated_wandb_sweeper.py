# import yaml
# import hydra
# import wandb
# from omegaconf import DictConfig


# def sweep_parameters(sweep_config ="configs/sweep_configs/sweepv1.yaml",project_name="param_sweeper"):
#     from train import ma1in
#     with open(sweep_config, 'r') as file:
#         try:
#             config = yaml.safe_load(file)
#             print(config)
#         except yaml.YAMLError as exc:
#             print(exc)
#     sweep_id = wandb.sweep(sweep=config, project=project_name)  
#     wandb.agent(sweep_id, function=main, count=10)
#     pass


# if __name__ == "__main__":
#     sweep_parameters("configs/SCEMILA_approaches/topo/topo_eucl_gray_image_image_input.yaml","param_sweeper")