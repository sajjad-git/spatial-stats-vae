# Uses Weights and Biases Sweep function to search for hyperparameters.
import os
import yaml
import argparse
import wandb
from wandb_train import run_training


def main():
    wandb.init(
        project='sweep-vae-loss-alphas-and-neural-layers',
        settings=wandb.Settings(_service_wait=300)
        )
    
    config = wandb.config
    #beta_max = 1 - (config.a_mse + config.a_content + config.a_style + config.a_spst) # beta is scheduled. it will go from 0.005 to beta_max
    run_training(config.epochs,
                  config.a_mse, config.a_content, config.a_style, config.a_spst, config.beta_max, 
                  config.content_layer, config.style_layer, 
                  learning_rate=config.learning_rate, fine_tune_lr=config.fine_tune_lr,
                  batch_size=config.batch_size, CNN_embed_dim=config.bottleneck_size,
                     schedule_KLD=config.schedule_KLD, schedule_spst=config.schedule_spst, dataset_name=config.dataset_name,
                     debugging=config.debugging) 
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Use W&B sweeps to sweep over hyperparameters. Put h-params in the sweep_config.yaml file.")
    parser.add_argument('--sweep_id', type=str, required=False, default=None, help="W&B sweep ID")
    args = parser.parse_args()
    
    # 1: login
    wandb.login()

    # 2: Load the YAML configuration file
    #with open(os.path.join(os.getcwd(), "src/models/config_files/test.yaml"), "r") as yaml_file:
    with open(os.path.join(os.getcwd(), "src/models/config_files/manual_config_spst_lr_test.yaml"), "r") as yaml_file:
        sweep_configuration = yaml.safe_load(yaml_file)
    
    # 3: Start the sweep
    if args.sweep_id != None:
        # provide a sweep id
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(
            sweep=sweep_configuration, 
            project='sweep-vae-loss-alphas-and-neural-layers',
            )
    print(sweep_id)
    wandb.agent(sweep_id, function=main, project='sweep-vae-loss-alphas-and-neural-layers')
