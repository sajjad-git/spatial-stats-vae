#%% Import the model and the data:
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import time
import csv

from resnet_vae import ResNet_VAE
from training_utils import seed_everything, reconstruct_images
from lines_dataset import LinesDataset
from utils import ThresholdTransform
from evaluate_outputs import threshold_image
from spatial_statistics_loss import TwoPointAutocorrelation, TwoPointSpatialStatsLoss
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

save_model_path = "/home/sajad/AI-generated-chemical-materials/models/resnetVAE__batch_similarity_loss+Falselr0.001bs32_a_spst_0_KLD_beta_1_spst_reduction_loss_sum_KLD_scheduled_False_spatial_stats_loss_scheduled_False_bottleneck_size_9_dataset_name_multiple_lines_seed_125"
epoch=400
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
vae.load_state_dict(torch.load(model_path, map_location=device))

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

autocorr_func = TwoPointAutocorrelation()
autocorr_loss = TwoPointSpatialStatsLoss(device, min_pixel_value=None, max_pixel_value=None)

#%% The plan
'''
    1. Print out 100 pairs of original-reconstruction to verify they have the correct orientation. For each pair,
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

# display 100 pairs of images
orig, recon, orig_autocorr, recon_autocorr = reconstruct_images(vae, train_loader, device, num_examples=100)

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
percent_diff_in_black_pixels = []

for i in tqdm(range(len(orig)), desc="Analyzing Images"):
    start_time = time.time()
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
    percent_diff_black_pixels = round(percent_diff_black_pixels, 4)
    percent_diff_in_black_pixels.append(percent_diff_black_pixels)
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

    # Iterate over training set
    for val_images, y in tqdm(train_loader, desc=f"Processing training Set for Image {i}"):
        for val_image in val_images:
            mse = F.mse_loss(reconstruction, val_image)
            mse_list.append(mse.item())
            if mse < min_mse:
                min_mse = mse
                most_similar_image = val_image

    # d. Average and Std of MSEs with training Set
    avg_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    msg = f"Average MSE: {avg_mse}, Standard Deviation of MSEs: {std_mse}\n"
    log_messages += msg
    print(msg)

    # e. Spatial statistics comparison
    spatial_stats_mse_list = []
    min_spatial_stats_mse = float('inf')
    most_similar_spatial_stats_image = None

    # Iterate over training set with tqdm for spatial statistics
    for val_images, y in tqdm(train_loader, desc=f"Processing Spatial Stats for Image {i}"):
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
    
    x_values = np.linspace(-orig_spst.shape[-1] // 2, orig_spst.shape[-1] // 2, orig_spst.shape[-1])
    y_values = np.linspace(-orig_spst.shape[-2] // 2, orig_spst.shape[-2] // 2, orig_spst.shape[-2])

    def save_plot(ax, image_name, fig, autocorr=None):
        image_dir = os.path.join(analysis_dir, f"sample_{i}")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        plot_filename = os.path.join(image_dir, f'{image_name}_{i}.pdf')
        
        if autocorr is not None:
            _, mini_ax = plt.subplots()
            im = mini_ax.imshow(autocorr.squeeze().cpu().numpy(), origin='lower', interpolation='none', cmap='gray', extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]])
            plt.colorbar(im, ax=mini_ax)
            plt.savefig(plot_filename)
            plt.close()
        else:
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(plot_filename, bbox_inches=extent.expanded(1.1, 1))


    # Display original and reconstructed images side by side
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(original.squeeze().cpu().numpy(), cmap='gray')
    ax1.axis('off')
    save_plot(ax1, "original_image", fig)
    ax1.set_title(f'Original Image Sample {i}')

    # Reconstructed Image
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(threshold_image(reconstruction).squeeze().cpu().numpy(), cmap='gray')
    ax2.axis('off')
    save_plot(ax2, "reconstructed_image", fig)
    ax2.set_title('Reconstructed Image')

    # Original Spatial Stats
    ax3 = fig.add_subplot(2, 4, 5)
    orig_spst = orig_autocorr[i]
    im = ax3.imshow(orig_spst.squeeze().cpu().numpy(), origin='lower', interpolation='none', cmap='gray', extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]])
    plt.colorbar(im, ax=ax3)
    save_plot(ax3, "reconstructed_image_autocorrelation", fig, autocorr=orig_spst)
    ax3.set_title('Original Spatial Stats')

    # Reconstructed Spatial Stats
    ax4 = fig.add_subplot(2, 4, 6)
    im = ax4.imshow(recon_spst.squeeze().cpu().numpy(), origin='lower', interpolation='none', cmap='gray', extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]])
    plt.colorbar(im, ax=ax4)
    save_plot(ax4, "original_image_autocorrelation", fig, autocorr=recon_spst)
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
    save_histogram(percent_diff_in_black_pixels, "Volume fraction difference between input and reconstruction", "volume_fraction_difference")
    # end plotting -------------------------------------------------------

    # save the histogram data
    # Define the file name
    filename = os.path.join(analysis_dir, "analysis_histogram_data.csv")

    # Writing to CSV
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Writing headers
        writer.writerow(["MSE Input-Recon Means", "MSE Input-Recon STDs", "Spatial Stats MSE Means", "Spatial Stats MSE STDs", "Percent Diff in Black Pixels"])

        # Writing data
        for i in range(len(mse_input_recon_means)):
            writer.writerow([
                mse_input_recon_means[i], 
                mse_input_recon_stds[i], 
                spatial_stats_mse_means[i], 
                spatial_stats_mse_stds[i], 
                percent_diff_in_black_pixels[i]
            ])
    print(f"Data saved to {filename}")
    
    # record the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

pdf.close()
log_file_path = os.path.join(analysis_dir, 'analysis_log.txt')
with open(log_file_path, 'w') as log_file:
    log_file.write(log_messages)
print("DONE")    


#%%
"""
For 5 reconstructions, 
    - Get the top 5 closest images from the training set in the data space
    - Get the top 5 closest images from the validation set in the data space
    - Get the top 5 closest images from the training set in the spatial statistics space
    - Get the top 5 closest images from the validation set in the spatial statistics space

    Save the individual MSEs 
"""

orig, recon, orig_autocorr, recon_autocorr = reconstruct_images(vae, train_loader, device, num_examples=5)

def get_top_5_closest_images(loader, target_image, spatial_stats=False):
    top_5_mses = [(float('inf'), None)] * 5  # Initialize with high MSEs

    for images, _ in tqdm(loader, desc="Processing"):
        for image in images:
            if spatial_stats:
                image_feature = autocorr_func.forward(image)
                target_feature = autocorr_loss.calculate_two_point_autocorr_pytorch(target_image.unsqueeze(0)).squeeze(0)
            else:
                image_feature = image
                target_feature = target_image

            mse = F.mse_loss(target_feature, image_feature).item()

            # Check if current MSE is lower than the highest in the top 5
            if mse < top_5_mses[-1][0]:
                top_5_mses[-1] = (mse, image)
                top_5_mses.sort()

    return top_5_mses

import matplotlib.pyplot as plt
import torchvision

def save_image_row(recon, images, filename):
    # Concatenate the top 5 images in a row
    images = [recon]+[images[i][1] for i in range(len(images))]
    concatenated_images = torchvision.utils.make_grid(images, nrow=6, padding=1, normalize=True)

    # Convert to numpy and transpose axes for plotting
    np_images = concatenated_images.numpy().transpose((1, 2, 0))

    # Plotting and saving
    plt.figure(figsize=(10, 2))
    plt.imshow(np_images)
    plt.xticks([])  # Hide x-axis ticks
    plt.yticks([])
    plt.xlabel("From the left: the reconstructed image, followed by the top 5 most similar images in decreasing order.")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# Iterate over each pair of original and reconstructed images
for i in range(len(orig)):
    print(f"Processing reconstruction {i}")
    reconstruction = recon[i]

    # Get top 5 images from the training and validation sets in the data space
    top_5_train_data_space = get_top_5_closest_images(train_loader, reconstruction)
    top_5_valid_data_space = get_top_5_closest_images(valid_loader, reconstruction)

    # Get top 5 images from the training and validation sets in the spatial statistics space
    top_5_train_spatial_stats = get_top_5_closest_images(train_loader, reconstruction, spatial_stats=True)
    top_5_valid_spatial_stats = get_top_5_closest_images(valid_loader, reconstruction, spatial_stats=True)
    
    # Save images
    save_image_row(reconstruction, top_5_train_data_space, os.path.join(analysis_dir, f'top_5_train_data_space_image_{i}.pdf'))
    save_image_row(reconstruction, top_5_valid_data_space, os.path.join(analysis_dir, f'top_5_valid_data_space_image_{i}.pdf'))
    save_image_row(reconstruction, top_5_train_spatial_stats, os.path.join(analysis_dir, f'top_5_train_spatial_stats_image_{i}.pdf'))
    save_image_row(reconstruction, top_5_valid_spatial_stats, os.path.join(analysis_dir, f'top_5_valid_spatial_stats_image_{i}.pdf'))

# %% Gather data to compare the histograms of the training against validation mse.
import os
import time
import numpy as np
import torch.nn.functional as F
import csv
from tqdm import tqdm

# Specify your interval here (e.g., every 100 iterations)
num_samples = 1000
save_interval = 10
log_messages = ""
loader_names = ['training', 'validation']
for i, loader in enumerate([train_loader, valid_loader]):
    loader_name = loader_names[i]
    orig, recon, orig_autocorr, recon_autocorr = reconstruct_images(vae, loader, device, num_examples=num_samples)
    analysis_dir = os.path.join(save_model_path, f'analysis_of_{loader_name}_data')

    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # Prepare CSV file for writing
    csv_filename = os.path.join(analysis_dir, f"analysis_of_{loader_name}_data.csv")
    
    # Check if CSV file exists and count the number of rows
    row_count = 0
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r') as file:
            reader = csv.reader(file)
            row_count = sum(1 for row in reader)

    # If file does not exist or has only header, start from beginning and write header
    if row_count <= 1:
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["MSE Input-Recon Means", "MSE Input-Recon STDs", "Spatial Stats MSE Means", "Spatial Stats MSE STDs"])

    # Initialize lists for data accumulation
    mse_input_recon_means = []
    mse_input_recon_stds = []
    spatial_stats_mse_means = []
    spatial_stats_mse_stds = []

    start_index = max(row_count - 1, 0) # Adjust start index based on existing data

    for i in tqdm(range(start_index, len(orig)), desc=f"Analyzing {loader_name} images"):
        start_time = time.time()
        # Safely squeeze tensors to handle dimensionality
        original = orig[i]
        reconstruction = recon[i]
        orig_spst = orig_autocorr[i]
        recon_spst = autocorr_loss.calculate_two_point_autocorr_pytorch(reconstruction.unsqueeze(0)).squeeze(0)

        # b. MSE between Input and Reconstruction
        mse_input_recon = F.mse_loss(original.view(-1), reconstruction.view(-1))
        msg = f"MSE between input and reconstruction for image {i}: {mse_input_recon}\n"
        log_messages += msg

        # c. MSE between Spatial Statistics
        mse_spatial_stats = F.mse_loss(orig_spst.view(-1), recon_spst.view(-1))
        msg = f"MSE between spatial statistics for image {i}: {mse_spatial_stats}\n"
        log_messages += msg

        # d, e, f - Initialization
        mse_list = []
        min_mse = float('inf')
        most_similar_image = None

        # Iterate over training set
        print(f"Processing {loader_name} set for mse.")
        for batch, y in loader:
            for image in batch:
                mse = F.mse_loss(reconstruction, image)
                mse_list.append(mse.item())
                if mse < min_mse:
                    min_mse = mse
                    most_similar_image = image

        # d. Average and Std of MSEs with training Set
        avg_mse = np.mean(mse_list)
        std_mse = np.std(mse_list)
        msg = f"Average MSE: {avg_mse}, Standard Deviation of MSEs: {std_mse}\n"
        log_messages += msg

        # e. Spatial statistics comparison
        spatial_stats_mse_list = []
        min_spatial_stats_mse = float('inf')
        most_similar_spatial_stats_image = None

        # Iterate over training set with tqdm for spatial statistics
        for batch, y in loader:
            for image in batch:
                image_spatial_stats = autocorr_func.forward(image)
                mse_spatial_stats = F.mse_loss(recon_spst, image_spatial_stats)
                spatial_stats_mse_list.append(mse_spatial_stats.item())
                if mse_spatial_stats < min_spatial_stats_mse:
                    min_spatial_stats_mse = mse_spatial_stats
                    most_similar_spatial_stats_image = image

        # Calculate and print the average and std of spatial statistics MSEs
        avg_spatial_stats_mse = np.mean(spatial_stats_mse_list)
        std_spatial_stats_mse = np.std(spatial_stats_mse_list)
        msg = f"Average MSE for Spatial Statistics: {avg_spatial_stats_mse}, Std of MSEs: {std_spatial_stats_mse}\n"
        log_messages += msg

        # Append line separator for readability in logs
        log_messages += "-"*40 + "\n"

        # Store the means and standard deviations
        mse_input_recon_means.append(avg_mse)
        mse_input_recon_stds.append(std_mse)
        spatial_stats_mse_means.append(avg_spatial_stats_mse)
        spatial_stats_mse_stds.append(std_spatial_stats_mse)
        
        # Check if it's time to save the results
        if (i + 1) % save_interval == 0 or i == len(orig) - 1:
            with open(csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                for j in range(len(mse_input_recon_means)):
                    writer.writerow([
                        mse_input_recon_means[j], 
                        mse_input_recon_stds[j], 
                        spatial_stats_mse_means[j], 
                        spatial_stats_mse_stds[j],
                    ])

            # Clear lists after saving
            mse_input_recon_means.clear()
            mse_input_recon_stds.clear()
            spatial_stats_mse_means.clear()
            spatial_stats_mse_stds.clear()

    print(f"Data saved to {csv_filename}")

print("DONE")
# %%
"""
Currently, we'd get plausible-looking reconstructions whether we
optimize in data space or in spatial stats space.

That's because currently, when we optimize in data space, the encoding
contains both information about the the texture that defines the
material (to some extent) and information about the specific image.

So what we'd expect is that if we make the encoding smaller and
smaller, at some point there will not be enough hidden units to store
both the material information and the specific image data.

So we'd expect that for an encoding of size e where e is small enough
to get the effect I describe below and large enough for the encoding
to work in spatial stats space:
* When optimizing in spatial stats space, the reconstructions will
still look plausible
* When optimizing in spatial stats space reconstructions with
encodings that are slightly perturbed will still look plausible
* When optimizing in data space, the reconstructions might look
plausible, but slightly perturbed encodings would not look plausible.

So what we want is
Train in data space and also in spatial stats space for  for encoding
bottleneck size = 1, 2, 3, ... 9
Output reconstructions, and also reconstruction of slightly pertuberd
encodings for

Find the largest bottleneck size where the reconstructions of slightly
perturbed encodings when optimizing in data space look bad.

"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import time
import csv

from resnet_vae import ResNet_VAE
from training_utils import seed_everything, reconstruct_images
from lines_dataset import LinesDataset
from utils import ThresholdTransform
from evaluate_outputs import threshold_image
from spatial_statistics_loss import TwoPointAutocorrelation, TwoPointSpatialStatsLoss
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

for a_spst in [1, 0]:
    bottleneck_size = 1

    seed = 125
    save_model_path = f"/home/sajad/AI-generated-chemical-materials/models/resnetVAE__batch_similarity_loss+Falselr0.001bs32_a_spst_{a_spst}_KLD_beta_1_spst_reduction_loss_sum_KLD_scheduled_False_spatial_stats_loss_scheduled_False_bottleneck_size_{bottleneck_size}_dataset_name_multiple_lines_seed_{seed}"
    epoch=100
    seed_everything(seed)
    batch_size = 32
    CNN_embed_dim = bottleneck_size
    res_size = 224
    use_cuda = torch.cuda.is_available()   
    device = torch.device("cuda" if use_cuda else "cpu")
    analysis_dir = os.path.join(save_model_path, 'analysis')
    # Create the directory if it does not exist
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    model_path = os.path.join(save_model_path, f"model_epoch{epoch}.pth")
    vae = ResNet_VAE(CNN_embed_dim=CNN_embed_dim, device=device).to(device)
    vae.resnet.requires_grad_(False)
    vae.load_state_dict(torch.load(model_path, map_location=device))

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

    autocorr_func = TwoPointAutocorrelation()
    autocorr_loss = TwoPointSpatialStatsLoss(device, min_pixel_value=None, max_pixel_value=None)

    num_images = 20  # or len(orig) if it's not a fixed number

    #orig, recon, orig_autocorr, recon_autocorr = reconstruct_images(vae, train_loader, device, num_examples=num_images)
    orig, recon, orig_autocorr, recon_autocorr, p_recon, p_recon_autocorr = reconstruct_images(vae, train_loader, device, num_examples=num_images, should_test_vae_robustness=True, perturbation=0.3)


    # Create a PDF file to save all the plots
    with PdfPages(os.path.join(analysis_dir, 'combined_images.pdf')) as pdf:
        for i in range(num_images):
            fig = plt.figure(figsize=(40, 20))  # Adjust size as needed

            original = orig[i]
            reconstruction = recon[i]
            orig_spst = orig_autocorr[i]
            recon_spst = autocorr_loss.calculate_two_point_autocorr_pytorch(reconstruction.unsqueeze(0)).squeeze(0)

            p_reconstruction = p_recon[i]
            p_recon_spst = autocorr_loss.calculate_two_point_autocorr_pytorch(p_reconstruction.unsqueeze(0)).squeeze(0)

            x_values = np.linspace(-orig_spst.shape[-1] // 2, orig_spst.shape[-1] // 2, orig_spst.shape[-1])
            y_values = np.linspace(-orig_spst.shape[-2] // 2, orig_spst.shape[-2] // 2, orig_spst.shape[-2])

            # Original Image
            ax = fig.add_subplot(6, num_images, i + 1)
            ax.imshow(orig[i].squeeze().cpu().numpy(), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Original {i}')

            # Original Spatial Stats
            ax = fig.add_subplot(6, num_images, num_images + i + 1)
            im = ax.imshow(orig_autocorr[i].squeeze().cpu().numpy(), origin='lower', cmap='gray')
            plt.colorbar(im, ax=ax)
            ax.axis('off')
            ax.set_title(f'Orig Spatial Stats {i}')

            # Reconstructed Image
            ax = fig.add_subplot(6, num_images, 2 * num_images + i + 1)
            ax.imshow(recon[i].squeeze().cpu().numpy(), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Reconstructed {i}')

            # Reconstructed Spatial Stats
            ax = fig.add_subplot(6, num_images, 3 * num_images + i + 1)
            im = ax.imshow(autocorr_loss.calculate_two_point_autocorr_pytorch(recon[i].unsqueeze(0)).squeeze(0).squeeze().cpu().numpy(), origin='lower', cmap='gray', extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]])
            plt.colorbar(im, ax=ax)
            ax.axis('off')
            ax.set_title(f'Recon Spatial Stats {i}')

            # Perturbed Reconstructed Image
            ax = fig.add_subplot(6, num_images, 4 * num_images + i + 1)
            ax.imshow(p_recon[i].squeeze().cpu().numpy(), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Perturbed Recon {i}')

            # Perturbed Reconstructed Spatial Stats
            ax = fig.add_subplot(6, num_images, 5 * num_images + i + 1)
            im = ax.imshow(autocorr_loss.calculate_two_point_autocorr_pytorch(p_recon[i].unsqueeze(0)).squeeze(0).squeeze().cpu().numpy(), origin='lower', cmap='gray', extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]])
            plt.colorbar(im, ax=ax)
            ax.axis('off')
            ax.set_title(f'Perturbed Recon Spatial Stats {i}')

            # Save the current page of the PDF file
            pdf.savefig(fig)
            plt.close(fig)

# %%

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import time
import csv

from resnet_vae import ResNet_VAE
from training_utils import seed_everything, reconstruct_images
from lines_dataset import LinesDataset
from utils import ThresholdTransform
from evaluate_outputs import threshold_image
from spatial_statistics_loss import TwoPointAutocorrelation, TwoPointSpatialStatsLoss
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

data_dir = 'multiple_lines'
dataset_path = '/home/sajad/AI-generated-chemical-materials/data'
res_size = 224
seed = 125
epoch=100
seed_everything(seed)
batch_size = 32
use_cuda = torch.cuda.is_available()   
device = torch.device("cuda" if use_cuda else "cpu")

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
autocorr_func = TwoPointAutocorrelation()
autocorr_loss = TwoPointSpatialStatsLoss(device, min_pixel_value=None, max_pixel_value=None)

# %%

num_images=3000 # The entire validation set

bottleneck_sizes = [1, 2, 3, 5, 7, 9]
validation_loss_results = {'a_spst=0': [], 'a_spst=1': []}
for bottleneck_size in bottleneck_sizes:
    for a_spst in [0, 1]:
        save_model_path = f"/home/sajad/AI-generated-chemical-materials/models/resnetVAE__batch_similarity_loss+Falselr0.001bs32_a_spst_{a_spst}_KLD_beta_1_spst_reduction_loss_sum_KLD_scheduled_False_spatial_stats_loss_scheduled_False_bottleneck_size_{bottleneck_size}_dataset_name_multiple_lines_seed_{seed}"
        CNN_embed_dim = bottleneck_size
        analysis_dir = os.path.join(save_model_path, 'analysis')
        # Create the directory if it does not exist
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        model_path = os.path.join(save_model_path, f"model_epoch{epoch}.pth")
        vae = ResNet_VAE(CNN_embed_dim=CNN_embed_dim, device=device).to(device)
        vae.resnet.requires_grad_(False)
        vae.load_state_dict(torch.load(model_path, map_location=device))
        orig, recon, orig_autocorr, recon_autocorr = reconstruct_images(vae, valid_loader, device, num_examples=num_images)
        val_loss = 0
        for i in range(len(orig)):
            original = orig[i]
            reconstruction = recon[i]
            orig_spst = orig_autocorr[i]
            recon_spst = autocorr_loss.calculate_two_point_autocorr_pytorch(reconstruction.unsqueeze(0)).squeeze(0)
            val_loss += F.mse_loss(orig_spst, recon_spst).item()
        val_loss /= len(orig)
        validation_loss_results[f'a_spst={a_spst}'].append(val_loss)
# %%
# Plotting results
plt.figure(figsize=(10, 6))
for a_spst, losses in validation_loss_results.items():
    plt.plot(bottleneck_sizes, losses, label=('MSE Criterion' if a_spst=="a_spst=0" else "Spatial Statistics Criterion"))

plt.xlabel('Bottleneck Size')
plt.ylabel('Validation Error')
plt.title('Validation Error vs. Bottleneck Size')
plt.legend()
plt.savefig("validation_loss_vs_bottleneck_size.pdf")
plt.show()
# %%
plt.figure(figsize=(10, 6))
for a_spst, losses in validation_loss_results.items():
    plt.scatter(bottleneck_sizes, losses, label=('MSE Criterion' if a_spst == "a_spst=0" else "Spatial Statistics Criterion"))


plt.xlabel('Bottleneck Size')
plt.ylabel('Validation Error')
plt.title('Validation Error vs. Bottleneck Size')
plt.legend()
plt.savefig("validation_loss_vs_bottleneck_size.pdf")
plt.show()
# %%
import random
import torch

def get_random_images_with_matching_labels(dataloader, input_batch_images, input_batch_labels):
    """
    Retrieves a batch of random images from the dataset with the same labels as the input batch.

    :param dataloader: DataLoader for the dataset.
    :param input_batch_images: Tensor of images from the input batch.
    :param input_batch_labels: Tensor of labels from the input batch.
    :return: Tuple (new_batch_images, new_batch_labels), where new_batch_images contains random images with labels matching those in new_batch_labels.
    """
    # Convert DataLoader dataset to list for easier processing
    dataset_list = list(dataloader.dataset)

    new_batch_images = []
    new_batch_labels = []

    for label in input_batch_labels:
        # Filter dataset for images with the same label
        same_label_images = [item[0] for item in dataset_list if item[1] == label.item()]
        
        # Randomly select one image with the same label
        if same_label_images:
            random_image = random.choice(same_label_images)
            new_batch_images.append(random_image)
            new_batch_labels.append(label)

    # Convert lists to tensors
    new_batch_images_tensor = torch.stack(new_batch_images)
    new_batch_labels_tensor = torch.stack(new_batch_labels)

    return new_batch_images_tensor, new_batch_labels_tensor

# %%

# Fetch a single batch
batch_images, batch_labels = next(iter(train_loader))

# Get a new batch with random images having matching labels
new_batch_images, new_batch_labels = get_random_images_with_matching_labels(train_loader, batch_images, batch_labels)

# Plotting to visualize the original and new images
fig, axs = plt.subplots(2, len(batch_images), figsize=(15, 6))

for i in range(len(batch_images)):
    # Original batch
    axs[0, i].imshow(batch_images[i].permute(1, 2, 0))
    axs[0, i].set_title(f'Original Label: {batch_labels[i].item()}')
    axs[0, i].axis('off')

    # New batch with matching labels
    axs[1, i].imshow(new_batch_images[i].permute(1, 2, 0))
    axs[1, i].set_title(f'New Label: {new_batch_labels[i].item()}')
    axs[1, i].axis('off')

plt.show()
# %%
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import time
import csv

from resnet_vae import ResNet_VAE
from training_utils import seed_everything, reconstruct_images
from lines_dataset import LinesDataset
from utils import ThresholdTransform
from evaluate_outputs import threshold_image
from spatial_statistics_loss import TwoPointAutocorrelation, TwoPointSpatialStatsLoss
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
a_spst = 0
epoch=400
seed = 125
bottleneck_size = 9
save_model_path = f"/home/sajad/AI-generated-chemical-materials/models/resnetVAE__batch_similarity_loss+Falselr0.001bs32_a_spst_{a_spst}_KLD_beta_1_spst_reduction_loss_sum_KLD_scheduled_False_spatial_stats_loss_scheduled_False_bottleneck_size_{bottleneck_size}_dataset_name_multiple_lines_seed_{seed}"
# save_model_path = "/home/sajad/AI-generated-chemical-materials/models/resnetVAE_lr0.001bs32_a_spst_1_KLD_beta_1_spst_reduction_loss_sum_KLD_scheduled_False_spatial_stats_loss_scheduled_False_bottleneck_size_9_dataset_name_multiple_lines_seed_125"
seed_everything(seed)
batch_size = 32
CNN_embed_dim = bottleneck_size
res_size = 224
use_cuda = torch.cuda.is_available()   
device = torch.device("cuda" if use_cuda else "cpu")

model_path = os.path.join(save_model_path, f"model_epoch{epoch}.pth")
vae = ResNet_VAE(CNN_embed_dim=CNN_embed_dim, device=device).to(device)
vae.resnet.requires_grad_(False)
vae.load_state_dict(torch.load(model_path, map_location=device))

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

autocorr_func = TwoPointAutocorrelation()
autocorr_loss = TwoPointSpatialStatsLoss(device, min_pixel_value=None, max_pixel_value=None)

# Specify your interval here (e.g., every 100 iterations)
num_samples = 1000
log_messages = ""
loader_name = 'validation'


orig, recon, orig_autocorr, recon_autocorr = reconstruct_images(vae, valid_loader, device, num_examples=num_samples)
analysis_dir = os.path.join(save_model_path, f'analysis_of_{loader_name}_data')

if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

# Prepare CSV file for writing
csv_filename = os.path.join(analysis_dir, f"analysis_of_{loader_name}_data_pairwise_recon_loss.csv")
    
    
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["MSE Input-Recon Means", "MSE Input-Recon STDs", "Spatial Stats MSE Means", "Spatial Stats MSE STDs"])

recon_mses = []
recon_autocorr_mses = []

for i in tqdm(range(len(orig)), desc=f"Analyzing {loader_name} images"):
    # Safely squeeze tensors to handle dimensionality
    original = orig[i]
    reconstruction = recon[i]
    orig_spst = orig_autocorr[i]
    recon_spst = autocorr_loss.calculate_two_point_autocorr_pytorch(reconstruction.unsqueeze(0)).squeeze(0)

    # b. MSE between Input and Reconstruction
    mse_input_recon = F.mse_loss(original.view(-1), reconstruction.view(-1))
    recon_mses.append(mse_input_recon.item())

    # c. MSE between Spatial Statistics
    mse_spatial_stats = F.mse_loss(orig_spst.view(-1), recon_spst.view(-1))
    recon_autocorr_mses.append(mse_spatial_stats.item())

    
# Check if it's time to save the results
with open(csv_filename, 'a', newline='') as file:
    writer = csv.writer(file)
    for j in range(len(orig)):
        writer.writerow([
            recon_mses[j], 
            0, 
            recon_autocorr_mses[j], 
            0,
        ])

print(f"Data saved to {csv_filename}")

print("DONE")
# %%
