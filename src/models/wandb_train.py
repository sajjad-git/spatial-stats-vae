import os
import time
import argparse
import wandb
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from resnet_vae import ResNet_VAE
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../data')
from make_circles_squares_dataset import CustomDataset
from utils import ThresholdTransform, check_mkdir
from training_utils import train, validation, MaterialSimilarityLoss, ExponentialScheduler, LossCoefficientScheduler, learning_rate_switcher, get_learning_rate, seed_everything, generate_from_noise, generate_reconstructions


def run_training(epochs, a_mse, a_content, a_style, a_spst, beta, content_layer, style_layer,
                learning_rate=1e-3, batch_size=32, CNN_embed_dim=256,
                  dropout_p=0.2, log_interval=2, save_interval=20, resume_training=False, last_epoch=0):
    seed=110
    seed_everything(seed)
    
    save_dir = os.path.join(os.getcwd(), "models")
    run_name = "resnetVAE_shapesData_" + f"lr{learning_rate}" + f"bs{batch_size}" + "_a_mse" + str(a_mse) + "_a_content" + str(a_content) + "_a_style" + str(a_style) + "_a_spst" + str(a_spst) + "_" + "content_layer" + f"{content_layer}" + "_" + "style_layer" + f"{style_layer}" + "_sum_reduction"
    save_model_path = os.path.join(save_dir, run_name)
    check_mkdir(save_model_path)    

    # alternatively, you could save in W&B but depending on the network speed, uploading the models can be slow.
    #save_model_path = wandb.run.dir

    # Detect devices
    use_cuda = torch.cuda.is_available()   
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print("Using", torch.cuda.device_count(), "GPU!")
    else:
        print("Training on CPU!")

    # Load Data
    res_size = 224
    # Define the transformation to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the pixel values to the range [-1, 1]
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Resize([res_size, res_size], antialias=True),
        ThresholdTransform(thr_255=240),
    ])
    dataset = CustomDataset(os.path.join(os.getcwd(), 'data/raw/labels.csv'), os.path.join(os.getcwd(), 'data/raw/shape_images'), transform)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.7), int(len(dataset)) - int(len(dataset)*0.7)])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # EncoderCNN architecture
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
    # Build model
    resnet_vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim, device=device).to(device)
    resnet_vae.resnet.requires_grad_(False)
    wandb.watch(resnet_vae)
    model_params = list(resnet_vae.parameters())
    optimizer = torch.optim.Adam(model_params, lr=learning_rate)
    beta_scheduler = ExponentialScheduler(start=0.005, max_val=beta, epochs=epochs) # start = 256/(224*224) = (latent space dim)/(input dim)
    loss_function = MaterialSimilarityLoss(device, content_layer=content_layer, style_layer=style_layer)
    a_spst_scheduler = LossCoefficientScheduler(a_spst, epochs)

    print({
        "seed": seed,
        "run_name": run_name, 
        #"content_layer_coeffs": loss_function.content_layer_coefficients,
        #"style_layer_coeffs": loss_function.style_layer_coefficients,
        })
    
    if resume_training:
        assert last_epoch != None
        resnet_vae.load_state_dict(torch.load(os.path.join(save_model_path,f'model_epoch{last_epoch}.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(save_model_path,f'optimizer_epoch{last_epoch}.pth')))
        print("Resuming pretrained model...")
    else:
        last_epoch = 0


    #start training
    for epoch in range(last_epoch, epochs):
        start = time.time()

        # get the scheduled KLD beta value
        # TODO: Test the effect of the scheduler on our results
        #beta = beta_scheduler.get_beta(epoch)
        beta=1

        # train, test model
        X_train, y_train, z_train, mu_train, logvar_train, training_losses = train(log_interval, resnet_vae, loss_function, device, train_loader, optimizer, epoch, save_model_path, a_mse, a_content, a_style, a_spst, beta)
        X_test, y_test, z_test, mu_test, logvar_test, validation_losses = validation(resnet_vae, loss_function, device, valid_loader, a_mse, a_content, a_style, a_spst, beta)
        mse_training_loss, content_training_loss, style_training_loss, spst_training_loss, kld_training_loss, overall_training_loss = training_losses
        mse_loss, content_loss, style_loss, spst_loss, kld_loss, overall_loss = validation_losses
        
        # increase spst loss value
        a_spst = a_spst_scheduler.step()
        a_mse = 1 - a_spst

        metrics = {
            "mse_training_loss": mse_training_loss, 
            "mse_validation_loss": mse_loss, 
            "spatial_stats_training_loss": spst_training_loss,
            "spatial_stats_validation_loss": spst_loss,
            "KLD_training_loss": kld_training_loss,
            "KLD_validation_loss": kld_loss,
            "overall_training_loss": overall_training_loss,
            "overall_validation_loss": overall_loss,
            "content_validation_loss": content_loss, 
            "content_training_loss": content_training_loss, 
            "style_training_loss": style_training_loss,
            "style_validation_loss": style_loss,
            "mu_training": mu_train,
            "mu_test": mu_test,
            "logvar_train": logvar_train,
            "logvar_test": logvar_test,
            }
        wandb.log(metrics)
        
        if (epoch+1)%save_interval==0:
            torch.save(resnet_vae.state_dict(), os.path.join(save_model_path, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
            torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
            np.save(os.path.join(save_model_path, 'X_train_epoch{}.npy'.format(epoch + 1)), X_train) #save last batch
            np.save(os.path.join(save_model_path, 'y_train_epoch{}.npy'.format(epoch + 1)), y_train)
            np.save(os.path.join(save_model_path, 'z_train_epoch{}.npy'.format(epoch + 1)), z_train)
            
            grid = generate_reconstructions(resnet_vae, device, X_train, z_train)
            print("Training figures generated succesfully.")
            imgs = wandb.Image(grid, caption='images together top: orig, bottom: recon')
            wandb.log({'Training reconstructions': imgs})
            print("Training reconstructions logged succesfully.")

            grid = generate_reconstructions(resnet_vae, device, X_test, z_test)
            print("Validation figures generated succesfully.")
            imgs = wandb.Image(grid, caption='images together top: orig, bottom: recon')
            wandb.log({'Validation reconstructions': imgs})
            print("Validation reconstructions logged succesfully.")
            
            try:
                grid = generate_from_noise(resnet_vae, device, 32).cpu()
                imgs = wandb.Image(grid)
                wandb.log({'Validation generated images from noise': imgs})
                print("Images generated from noise successfully.")
            except Exception as e:
                print(f"Error generating images from noise: {e}")

        print(f"epoch time elapsed {time.time() - start} seconds")
        print("-------------------------------------------------")

        # if epoch > 0:
        #     break

    print(f"Finished training for {run_name}.")



