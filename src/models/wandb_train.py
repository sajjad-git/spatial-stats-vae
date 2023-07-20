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
from training_utils import train, validation, MaterialSimilarityLoss


def run_training(epochs, a_content, a_style, a_spst, beta, content_layer, style_layer,
                learning_rate=1e-3, batch_size=32, CNN_embed_dim=256,
                  dropout_p=0.2, log_interval=2, save_interval=10, resume_training=False, last_epoch=None):

    save_dir = os.path.join(os.getcwd(), "models")
    save_model_path = "resnetVAE_shapesData_" + f"lr{learning_rate}" + f"bs{batch_size}" + "_loss_content" + str(a_content) + "_style" + str(a_style) + "_spst" + str(a_spst) + "_" + "content_layer" + f"{content_layer}" + "_" + "style_layer" + f"{style_layer}"
    save_model_path = os.path.join(save_dir, save_model_path)
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
        transforms.Resize([res_size, res_size]),
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
    model_params = list(resnet_vae.parameters())
    optimizer = torch.optim.Adam(model_params, lr=learning_rate)
    loss_function = MaterialSimilarityLoss(device, content_layer=content_layer, style_layer=style_layer)
    wandb.watch(resnet_vae)
    
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
        # train, test model
        X_train, y_train, z_train, mu_train, logvar_train, train_loss = train(log_interval, resnet_vae, loss_function, device, train_loader, optimizer, epoch, save_model_path, a_content, a_style, a_spst, beta)
        X_test, y_test, z_test, mu_test, logvar_test, test_loss = validation(resnet_vae, loss_function, device, valid_loader, a_content, a_style, a_spst, beta)
        
        # test_loss = [0,1,0.5,2,5]
        content_loss, style_loss, spst_loss, kld_loss, overall_loss = test_loss
        metrics = {"content_loss": content_loss, 
               "style_loss": style_loss,
                "spatial_stats_loss": spst_loss,
                "KLD_loss": kld_loss,
                "overall_loss": overall_loss}
        
        wandb.log(metrics)
        # X_train = np.array([0, 0])
        # y_train = np.array([0, 0])
        # z_train = np.array([0, 0])
        if (epoch+1)%10==0:
            torch.save(resnet_vae.state_dict(), os.path.join(save_model_path, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
            torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
            print("Epoch {} model saved!".format(epoch + 1))
            np.save(os.path.join(save_model_path, 'X_train_epoch{}.npy'.format(epoch + 1)), X_train) #save last batch
            np.save(os.path.join(save_model_path, 'y_train_epoch{}.npy'.format(epoch + 1)), y_train)
            np.save(os.path.join(save_model_path, 'z_train_epoch{}.npy'.format(epoch + 1)), z_train)
        print(f"epoch time elapsed {time.time() - start} seconds")
        print("-------------------------------------------------")
    
    print("Finished training!")



