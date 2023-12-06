#%% Import the model and the data:
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from resnet_vae import ResNet_VAE
from training_utils import seed_everything, reconstruct_images
from lines_dataset import LinesDataset
from utils import ThresholdTransform
from evaluate_outputs import threshold_image
from spatial_statistics_loss import TwoPointAutocorrelation, TwoPointSpatialStatsLoss
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
save_model_path = "/home/sajad/AI-generated-chemical-materials/models/resnetVAE_lr0.001bs32_a_spst_1_KLD_beta_1_spst_reduction_loss_sum_KLD_scheduled_True_spatial_stats_loss_scheduled_False_bottleneck_size_9_dataset_name_multiple_lines_seed_127"
save_model_path = "/home/sajad/AI-generated-chemical-materials/models/resnetVAE_lr0.001bs32_a_spst_1_KLD_beta_1_spst_reduction_loss_sum_KLD_scheduled_False_spatial_stats_loss_scheduled_False_bottleneck_size_9_dataset_name_multiple_lines_seed_125"
epoch=1500
seed = 125
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

data_dir = 'multiple_lines'
dataset_path = '/home/sajad/AI-generated-chemical-materials/data'
transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to tensor
        transforms.Normalize((0.5,), (0.5,)), 
        transforms.Resize([res_size, res_size], antialias=True),
        ThresholdTransform(thr_255=240),
    ])
dataset = LinesDataset(f'{dataset_path}/{data_dir}/labels.csv', f'{dataset_path}/{data_dir}/images', transform)
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.7), int(len(dataset)) - int(len(dataset)*0.7)])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# display 200 pairs of images
orig, recon, orig_autocorr, recon_autocorr = reconstruct_images(vae, train_loader, device, num_examples=1)
#orig, recon, orig_autocorr, recon_autocorr = reconstruct_images(vae, valid_loader, device, num_examples=10)

autocorr_func = TwoPointAutocorrelation()
autocorr_loss = TwoPointSpatialStatsLoss(device, min_pixel_value=None, max_pixel_value=None)

#%% The plan
'''
    1. Print out 200 pairs of original-reconstruction to verify they have the correct orientation. For each pair,
        a. Print out the percent difference in length and width of pixels of the original and reconstructions, after passing reconstructed image 
        through threshold_image filter.
        b. Print the MSE between the input and the reconstruction (we want to show that this is large).
        c. Print the MSE between the spatial statistis of the input and the reconstrction (we want to say that this is small).
        d. Print the avg. MSE and the std of MSEs of the reconstruction and every example in the training set.
        e. Print the avg. MSE and the std of MSEs of the spatial statistics of the reconstruction and the spatial statistics 
        every example in the training set.
        f. Also, take the smallest MSE of reconstruction and training example, and display the reconstruction and the training example (this 
        would be the most similar image).
'''
# Analysis Plan Implementation

# Initialize an empty string for logging
log_messages = ""  
analysis_dir = os.path.join(save_model_path, 'analysis')

# Create the analysis directory if it doesn't exist
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

# PDF file for the output
pdf_file_path = os.path.join(analysis_dir, 'analysis_results.pdf')
pdf = PdfPages(pdf_file_path)

mse_input_recon_means = []
mse_input_recon_stds = []
spatial_stats_mse_means = []
spatial_stats_mse_stds = []

for i in tqdm(range(len(orig)), desc="Analyzing Images"):
    # Safely squeeze tensors to handle dimensionality
    original = orig[i]
    reconstruction = recon[i]
    orig_spst = orig_autocorr[i]
    recon_spst = autocorr_loss.calculate_two_point_autocorr_pytorch(reconstruction.unsqueeze(0)).squeeze(0)

    # a. Percent Difference in Black Pixels
    thresholded_recon = threshold_image(reconstruction)
    original_black_pixels = torch.sum(original == 0).item()
    thresholded_black_pixels = torch.sum(thresholded_recon == 0).item()
    percent_diff_black_pixels = abs(original_black_pixels - thresholded_black_pixels) / original_black_pixels * 100
    msg = f"Percent difference in black pixels for image {i}: {percent_diff_black_pixels}%\n"
    log_messages += msg
    print(msg)

    # b. MSE between Input and Reconstruction
    mse_input_recon = F.mse_loss(original.view(-1), reconstruction.view(-1))
    mse = f"MSE between input and reconstruction for image {i}: {mse_input_recon}\n"
    log_messages += msg
    print(msg)

    # c. MSE between Spatial Statistics
    mse_spatial_stats = F.mse_loss(orig_spst.view(-1), recon_spst.view(-1))
    msg = f"MSE between spatial statistics for image {i}: {mse_spatial_stats}\n"
    log_messages += msg
    print(msg)

    # d, e, f - Initialization
    mse_list = []
    min_mse = float('inf')
    most_similar_image = None

    # Iterate over validation set
    for val_images, y in tqdm(valid_loader, desc=f"Processing Validation Set for Image {i}"):
        for val_image in val_images:
            mse = F.mse_loss(reconstruction, val_image)
            mse_list.append(mse.item())
            if mse < min_mse:
                min_mse = mse
                most_similar_image = val_image

    # d. Average and Std of MSEs with Validation Set
    avg_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    msg = f"Average MSE: {avg_mse}, Standard Deviation of MSEs: {std_mse}\n"
    log_messages += msg
    print(msg)

    # e. Spatial statistics comparison
    spatial_stats_mse_list = []
    min_spatial_stats_mse = float('inf')
    most_similar_spatial_stats_image = None

    # Iterate over validation set with tqdm for spatial statistics
    for val_images, y in tqdm(valid_loader, desc=f"Processing Spatial Stats for Image {i}"):
        for val_image in val_images:
            val_spatial_stats = autocorr_func.forward(val_image)
            mse_spatial_stats = F.mse_loss(recon_spst, val_spatial_stats)
            spatial_stats_mse_list.append(mse_spatial_stats.item())
            if mse_spatial_stats < min_spatial_stats_mse:
                min_spatial_stats_mse = mse_spatial_stats
                most_similar_spatial_stats_image = val_image

    # Calculate and print the average and std of spatial statistics MSEs
    avg_spatial_stats_mse = np.mean(spatial_stats_mse_list)
    std_spatial_stats_mse = np.std(spatial_stats_mse_list)
    msg = f"Average MSE for Spatial Statistics: {avg_spatial_stats_mse}, Std of MSEs: {std_spatial_stats_mse}\n"
    log_messages += msg
    print(msg)

    # Append line separator for readability in logs
    log_messages += "-"*40 + "\n"

    # Store the means and standard deviations
    mse_input_recon_means.append(avg_mse)
    mse_input_recon_stds.append(std_mse)
    spatial_stats_mse_means.append(avg_spatial_stats_mse)
    spatial_stats_mse_stds.append(std_spatial_stats_mse)

    # plot the results
    # -------------------------------------------------------
    fig = plt.figure(figsize=(40, 20)) 
    
    def save_plot(ax, image_name, fig):
        image_dir = os.path.join(analysis_dir, f"sample_{i}")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        plot_filename = os.path.join(image_dir, f'{image_name}_{i}.pdf')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(plot_filename, bbox_inches=extent.expanded(1.1, 1.1))

    # Display original and reconstructed images side by side
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(original.squeeze().cpu().numpy(), cmap='gray')
    ax1.axis('off')
    save_plot(ax1, "original_image", fig)
    ax1.set_title('Original Image')

    # Reconstructed Image
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(threshold_image(reconstruction).squeeze().cpu().numpy(), cmap='gray')
    ax2.axis('off')
    save_plot(ax2, "reconstructed_image", fig)
    ax2.set_title('Reconstructed Image')

    # Original Spatial Stats
    ax3 = fig.add_subplot(2, 4, 5)
    orig_spst = orig_autocorr[i]
    x_values = np.linspace(-orig_spst.shape[-1] // 2, orig_spst.shape[-1] // 2, orig_spst.shape[-1])
    y_values = np.linspace(-orig_spst.shape[-2] // 2, orig_spst.shape[-2] // 2, orig_spst.shape[-2])
    im = ax3.imshow(orig_spst.squeeze().cpu().numpy(), origin='lower', interpolation='none', cmap='gray', extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]])
    plt.colorbar(im, ax=ax3)
    save_plot(ax3, "reconstructed_image_autocorrelation", fig)
    ax3.set_title('Original Spatial Stats')

    # Reconstructed Spatial Stats
    ax4 = fig.add_subplot(2, 4, 6)
    im = ax4.imshow(recon_spst.squeeze().cpu().numpy(), origin='lower', interpolation='none', cmap='gray', extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]])
    plt.colorbar(im, ax=ax4)
    save_plot(ax4, "original_image_autocorrelation", fig)
    ax4.set_title('Reconstructed Spatial Stats')

    # f. Display and save most similar image
    # Most Similar Image
    ax5 = fig.add_subplot(2, 4, 3)
    ax5.imshow(most_similar_image.squeeze().cpu().numpy(), cmap='gray')
    ax5.axis('off')
    save_plot(ax5, "most_similar_image", fig)
    ax5.set_title('Most Similar Image')

    # Most Similar Spatial Stats Image
    ax6 = fig.add_subplot(2, 4, 7)
    ax6.imshow(most_similar_spatial_stats_image.squeeze().cpu().numpy(), cmap='gray')
    ax6.axis('off')
    save_plot(ax6, "most_similar_image_based_on_autocorrelation", fig)
    ax6.set_title('Most Similar Image Based On Spatial Statistics')

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.show()
    plt.close(fig)
    
    # save histograms of means and stds
    def save_histogram(data, title, filename):
        plt.figure()
        plt.hist(data, bins=20, color='blue', alpha=0.7)
        #plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(analysis_dir, f'{filename}.pdf'))
        plt.close()

    # Plot and save the histograms
    save_histogram(mse_input_recon_means, 'MSE Input-Reconstruction Means', 'mse_input_recon_means')
    save_histogram(mse_input_recon_stds, 'MSE Input-Reconstruction Standard Deviations', 'mse_input_recon_stds')
    save_histogram(spatial_stats_mse_means, 'Spatial Stats MSE Means', 'spatial_stats_mse_means')
    save_histogram(spatial_stats_mse_stds, 'Spatial Stats MSE Standard Deviations', 'spatial_stats_mse_stds')

    # end plotting -------------------------------------------------------

pdf.close()
log_file_path = os.path.join(analysis_dir, 'analysis_log.txt')
with open(log_file_path, 'w') as log_file:
    log_file.write(log_messages)
print("DONE")    

# %% Generating the plots
"""
    Generate the following:
        1. Two rows of (original, reconstruction, spatial stats of the original, spatial stats of the reconstruction).
            - with colorbar on the spatial stats heatmap.

"""
