import numpy as np
import torch
import sys
import os
from PIL import Image

def stack_and_save_images(dir_path, epoch):
    # Hardcoded image names
    img_names = [f'original_images_epoch{epoch}.npy', f'reconstructed_images_epoch{epoch}.npy', f'original_autocorr_epoch{epoch}.npy', f'reconstructed_autocorr_epoch{epoch}.npy']
    imgs = [np.load(os.path.join(dir_path, name)) for name in img_names]

    # Convert to PyTorch tensors
    imgs = [torch.tensor(img, dtype=torch.float32) for img in imgs]

    # Stack images 1 and 2 vertically, and 3 and 4 vertically
    stacked1 = torch.cat([imgs[0], imgs[1]], dim=2) # Change axis if needed
    stacked2 = torch.cat([imgs[2], imgs[3]], dim=2) # Change axis if needed

    # Stack the two pairs horizontally
    final_img = torch.cat([stacked1, stacked2], dim=3) # Change axis if needed

    # Ensure 'results' directory exists
    results_dir = os.path.join(dir_path, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save each image in the batch
    for i in range(final_img.shape[0]):
        # Convert tensor to PIL Image for saving
        img = Image.fromarray(final_img[i].numpy().astype('uint8'))
        img.save(os.path.join(results_dir, f"result_{i}.jpg"))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_dir> <epoch>")
        sys.exit(1)

    path_to_images = sys.argv[1]
    epoch = sys.argv[2]
    stack_and_save_images(path_to_images, epoch)
