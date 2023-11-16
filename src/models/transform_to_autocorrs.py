import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from shapes_dataset import ShapesDataset
from lines_dataset import LinesDataset
from utils import ThresholdTransform
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from spatial_statistics_loss import calculate_2pt_autocorr
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torch
from tqdm import tqdm

def normalize(image_np, min_pixel_value, max_pixel_value):
    normalized_image = 255 * (image_np - min_pixel_value) / (max_pixel_value - min_pixel_value)
    normalized_image = normalized_image.clip(0, 255).astype(np.uint8)
    return normalized_image

def transform_and_save(dataset, transform, save_dir):
    max_pixel_value = -np.inf
    min_pixel_value = np.inf

    for idx, (image_data, _) in tqdm(enumerate(dataset), total=len(dataset)):
        if isinstance(image_data, str):
            image = Image.open(image_data)
        else:
            image = image_data

        transformed_image = transform(image)
        transformed_image = calculate_2pt_autocorr(transformed_image)

        if transformed_image.dim() == 3 and transformed_image.shape[0] == 1:  # If single channel image
            transformed_image = transformed_image.squeeze(0)

        transformed_image_np = transformed_image.numpy()

        # Update the max and min values
        image_max_pixel_value = transformed_image_np.max()
        image_min_pixel_value = transformed_image_np.min()
        max_pixel_value = max(max_pixel_value, image_max_pixel_value)
        min_pixel_value = min(min_pixel_value, image_min_pixel_value)

        # Normalize and convert back to PIL Image for saving
        #normalized_image_np = normalize(transformed_image_np, image_min_pixel_value, image_max_pixel_value)
        #normalized_image = Image.fromarray(normalized_image_np)
        #normalized_image.save(os.path.join(save_dir, f'autocorr_image_{idx}.png'))

    return max_pixel_value, min_pixel_value

def main():
    if len(sys.argv) != 2:
        dataset_name = 'lines'
    else:
        dataset_name = sys.argv[1]

    
    res_size = 224
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), 
        transforms.Resize([res_size, res_size], antialias=True),
        ThresholdTransform(thr_255=240),
    ])

    if dataset_name == 'lines':
        data_dir = 'lines'
        dataset = LinesDataset(os.path.join(os.getcwd(), f'data/{data_dir}/labels.csv'), os.path.join(os.getcwd(), f'data/{data_dir}/images'))
    elif dataset_name == 'shapes':
        data_dir = 'shapes'
        dataset = ShapesDataset(os.path.join(os.getcwd(), f'data/{data_dir}/labels.csv'), os.path.join(os.getcwd(), f'data/{data_dir}/images'))
    else:
        print(f"Dataset {dataset_name} not recognized.")
        sys.exit(1)

    save_dir = os.path.join(os.getcwd(), f'data/{data_dir}/2pt_autocorr_images')
    os.makedirs(save_dir, exist_ok=True)

    max_pixel_value, min_pixel_value = transform_and_save(dataset, transform, save_dir)

    with open(os.path.join(os.path.join(os.getcwd(), f'data/{data_dir}'), 'pixel_values.txt'), 'w') as f:
        f.write(f"Max pixel value: {max_pixel_value}\n")
        f.write(f"Min pixel value: {min_pixel_value}\n")

    print(f"Transformed images saved in {save_dir}")
    print(f"Max and Min pixel values saved in {save_dir}/pixel_values.txt")

if __name__ == "__main__":
    main()
