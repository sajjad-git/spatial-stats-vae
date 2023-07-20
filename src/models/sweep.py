# Uses Weights and Biases Sweep function to search for hyperparameters.
import os
import yaml
import argparse
import wandb
from wandb_train import run_training


def main():
    wandb.init(project='sweep-vae-loss-alphas-and-neural-layers')
    
    config = wandb.config
    beta = 1 - (config.a_content + config.a_style + config.a_spst)
    content_layer = f"conv_{config.content_layer}"
    style_layer = f"conv_{config.style_layer}"
    run_training(config.epochs, config.a_content, config.a_style, config.a_spst, beta, 
                     content_layer, style_layer) 
    
if __name__=="__main__":
    # parser = argparse.ArgumentParser(description="Use W&B sweeps to sweep over hyperparameters. Put h-params in the sweep_config.yaml file.")
    # parser.add_argument('--api_key', type=str, required=True, help="W&B api key.")
    # args = parser.parse_args()

    wandb.login()

    # Load the YAML configuration file
    #with open("sweep_config.yaml", "r") as yaml_file:
    with open(os.path.join(os.getcwd(), "src/models/test.yaml"), "r") as yaml_file:
        sweep_configuration = yaml.safe_load(yaml_file)
    
    # 3: Start the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='sweep-vae-loss-alphas-and-neural-layers'
        )
    wandb.agent(sweep_id, function=main)
