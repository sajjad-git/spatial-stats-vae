import os
import time
import argparse

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from resnet_vae import ResNet_VAE
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../data')
from shapes_dataset import CustomDataset
from utils import ThresholdTransform, check_mkdir
from training_utils import train, validation, MaterialSimilarityLoss


def run_training(epochs, a_content, a_style, a_spst, beta, content_layer, style_layer,
                  save_model_path, learning_rate=1e-3, batch_size=32, CNN_embed_dim=256,
                  dropout_p=0.2, log_interval=2, save_interval=10, resume_training=False, last_epoch=None):
    check_mkdir(save_model_path)

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

    # record training process
    epoch_train_losses, epoch_test_losses = [], []
    
    if resume_training:
        assert last_epoch != None
        resnet_vae.load_state_dict(torch.load(os.path.join(save_model_path,f'model_epoch{last_epoch}.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(save_model_path,f'optimizer_epoch{last_epoch}.pth')))
        epoch_train_losses = list(np.load(os.path.join(save_model_path, 'ResNet_VAE_training_loss.npy')))
        epoch_test_losses = list(np.load(os.path.join(save_model_path, 'ResNet_VAE_training_loss.npy')))
        print("Resuming pretrained model...")
    else:
        last_epoch = 0


    #start training
    for epoch in range(last_epoch, epochs):
        start = time.time()
        # train, test model
        X_train, y_train, z_train, mu_train, logvar_train, train_loss = train(log_interval, resnet_vae, loss_function, device, train_loader, optimizer, epoch, save_model_path, a_content, a_style, a_spst, beta)
        X_test, y_test, z_test, mu_test, logvar_test, test_loss = validation(resnet_vae, loss_function, device, valid_loader, a_content, a_style, a_spst, beta)

        epoch_train_losses.append(train_loss)
        epoch_test_losses.append(test_loss)
        
        if (epoch+1)%save_interval==0:
            torch.save(resnet_vae.state_dict(), os.path.join(save_model_path, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
            torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
            print("Epoch {} model saved!".format(epoch + 1))
            np.save(os.path.join(save_model_path, 'ResNet_VAE_training_loss.npy'), np.array(epoch_train_losses))
            np.save(os.path.join(save_model_path, 'ResNet_VAE_validation_loss.npy'), np.array(epoch_test_losses))
            np.save(os.path.join(save_model_path, 'X_train_epoch{}.npy'.format(epoch + 1)), X_train) #save last batch
            np.save(os.path.join(save_model_path, 'y_train_epoch{}.npy'.format(epoch + 1)), y_train)
            np.save(os.path.join(save_model_path, 'z_train_epoch{}.npy'.format(epoch + 1)), z_train)
        print(f"epoch time elapsed {time.time() - start} seconds")
        print("-------------------------------------------------")


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Script to run some generative model on some dataset.")
    """
    epochs, a_content, a_style, a_spst, beta, content_layer, style_layer,
                  save_model_path, learning_rate=1e-3, batch_size=32, CNN_embed_dim=256,
                  dropout_p=0.2, log_interval=2, save_interval=10, resume_training=False, last_epoch=None
    """
    # Add command-line arguments
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs to train the model for.")
    parser.add_argument('--a_content', type=float, default=0.05, help="alpha value for content loss of neural style transfer.")
    parser.add_argument('--a_style', type=float, default=0.15, help="alpha value for style loss of neural style transfer.")
    parser.add_argument('--a_spst', type=float, default=0.4, help="alpha value for spatial stats loss.")
    parser.add_argument('--content_layer', type=int, default=4, help="VGG19 layer used for extracting content features.")
    parser.add_argument('--style_layer', type=int, default=4, help="VGG19 layer used for extracting style features.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--bs', type=int, default=32, help="Batch size.")
    parser.add_argument('--dropout_p', type=float, default=0.2, help="Drop out rate.")
    parser.add_argument('--log_interval', type=int, default=2, help="Log interval")
    parser.add_argument('--save_interval', type=int, default=10, help="Every how many epochs to save model states.")
    parser.add_argument('--resume_training', type=bool, default=False, help="To resume a training job.")
    parser.add_argument('--last_epoch', type=int, help="Last training epoch save in files.")

    # Parse the command-line arguments
    args = parser.parse_args()
    save_dir = os.path.join(os.getcwd(), "models")
    save_model_path = "resnetVAE_shapesData_" + f"lr{args.lr}" + f"bs{args.bs}" + "_loss_content" + str(args.a_content) + "_style" + str(args.a_style) + "_spst" + str(args.a_spst) + "_" + "content_layer" + f"conv_{args.content_layer}" + "_" + "style_layer" + f"conv_{args.style_layer}" + "_" + f"dropout{args.dropout_p}" 
    save_model_path = os.path.join(save_dir, save_model_path)
    
    run_training(epochs=args.epochs, 
                a_content=args.a_content, 
                a_style=args.a_style, 
                a_spst=args.a_spst, 
                beta=1-(args.a_content+args.a_style+args.a_spst),
                content_layer=f"conv_{args.content_layer}",
                style_layer=f"conv_{args.style_layer}",
                save_model_path=save_model_path,
                learning_rate=args.lr,
                batch_size=args.bs,
                dropout_p=args.dropout_p,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                resume_training=args.resume_training,
                last_epoch=args.last_epoch,
                )


if __name__ == "__main__":
    main()
