import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse

from resnet_vae import ResNet_VAE
from training_utils import seed_everything, reconstruct_images
from lines_dataset import LinesDataset
from utils import ThresholdTransform

# Define main function
def main(save_model_path, epoch):
    # Your existing code here
    seed = 127
    seed_everything(seed)
    batch_size = 32
    CNN_embed_dim = 9
    res_size = 224
    use_cuda = torch.cuda.is_available()   
    device = torch.device("cuda" if use_cuda else "cpu")

    model_path = os.path.join(save_model_path, f"model_epoch{epoch}.pth")
    vae = ResNet_VAE(CNN_embed_dim=CNN_embed_dim, device=device).to(device)
    vae.resnet.requires_grad_(False)
    vae.load_state_dict(torch.load(model_path))

    data_dir = 'lines'
    dataset_path = "/Users/sajjad/Desktop/argonne/AI-generated-chemical-materials/data"
    transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to tensor
            transforms.Normalize((0.5,), (0.5,)), 
            transforms.Resize([res_size, res_size], antialias=True),
            ThresholdTransform(thr_255=240),
        ])
    dataset = LinesDataset(f'{dataset_path}/{data_dir}/labels.csv', f'{dataset_path}/{data_dir}/images', transform)
    _, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.7), int(len(dataset)) - int(len(dataset)*0.7)])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # save 100 pairs of images
    orig, recon, orig_autocorr, recon_autocorr = reconstruct_images(vae, valid_loader, device, num_examples=100)
    np.save(os.path.join(save_model_path, 'original_images_epoch{}.npy'.format(epoch)), orig.numpy())
    np.save(os.path.join(save_model_path, 'reconstructed_images_epoch{}.npy'.format(epoch)), recon.numpy())
    np.save(os.path.join(save_model_path, 'original_autocorr_epoch{}.npy'.format(epoch)), orig_autocorr.numpy())
    np.save(os.path.join(save_model_path, 'reconstructed_autocorr_epoch{}.npy'.format(epoch)), recon_autocorr.numpy())
    print("Original and reconstructed images and their autocorrelations saved successfully.")

if __name__ == "__main__":
    # Parse command line arguments
    #parser = argparse.ArgumentParser(description="Run VAE model reconstruction")
    #parser.add_argument("save_model_path", type=str, help="Path to save the model")
    #parser.add_argument("epoch", type=int, help="Epoch number")

    #args = parser.parse_args()
    save_model_path = "/Users/sajjad/Desktop/argonne/AI-generated-chemical-materials/models/resnetVAE_lr0.001bs2_a_spst_1_KLD_beta_1_spst_reduction_loss_sum_KLD_scheduled_False_spatial_stats_loss_scheduled_False_bottleneck_size_9_dataset_name_lines_seed_110"
    epoch=1
    # Call main function with parsed arguments
    #main(args.save_model_path, args.epoch)
    main(save_model_path, epoch)
