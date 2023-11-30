import numpy as np
import torch
import sys
import os
from PIL import Image
from evaluate_outputs import threshold_image

def stack_and_save_images(dir_path, epoch):
    # Hardcoded image names with dynamic epoch
    img_names = [
        f'original_images_epoch{epoch}.npy',
        f'reconstructed_images_epoch{epoch}.npy',
        f'original_autocorr_epoch{epoch}.npy',
        f'reconstructed_autocorr_epoch{epoch}.npy'
    ]
    imgs = [np.load(os.path.join(dir_path, name)) for name in img_names]

    # Convert to PyTorch tensors and remove singleton color channel dimension
    imgs = [torch.tensor(img, dtype=torch.float32).squeeze(1) for img in imgs]

    # Apply threshold to reconstructions
    imgs[1] = torch.stack([threshold_image(recon) for recon in imgs[1]])

    # Normalize images in batches 3 and 4
    imgs[2] = (imgs[2] - imgs[2].min()) / (imgs[2].max() - imgs[2].min())  # Normalize batch 3
    imgs[3] = torch.stack([(img - img.min()) / (img.max() - img.min()) for img in imgs[3]])  # Normalize batch 4
    
    # Stack images vertically and then horizontally
    stacked1 = torch.cat([imgs[0], imgs[1]], dim=1)  # Stacking along height
    stacked2 = torch.cat([imgs[2], imgs[3]], dim=1)  # Stacking along height
    final_img = torch.cat([stacked1, stacked2], dim=2)  # Stacking along width

    # Ensure 'results' directory exists
    results_dir = os.path.join(dir_path, 'image_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save each image in the batch
    for i in range(final_img.shape[0]):
        # Convert tensor to numpy array and rescale
        img_array = final_img[i].numpy() * 255  # Rescale to 0-255 range

        # Ensure correct data type
        img_array = img_array.astype(np.uint8)

        # Convert grayscale to RGB if needed
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)

        # Convert to PIL Image and save
        img = Image.fromarray(img_array)
        img.save(os.path.join(results_dir, f"result_epoch{epoch}_{i}.jpg"))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_dir> <epoch>")
        sys.exit(1)

    path_to_images = sys.argv[1]
    epoch = sys.argv[2]
    stack_and_save_images(path_to_images, epoch)
