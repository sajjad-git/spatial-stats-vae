import numpy as np
import torch
import sys
import os 
def threshold_image(image, threshold=0.05):
    """
    Apply thresholding to the image. Pixels with values below the threshold are set to zero,
    and pixels with values equal to or above the threshold are set to one.
    """
    return torch.where(image < threshold, torch.zeros_like(image), torch.ones_like(image))

def count_zero_pixels(image):
    """
    Count the number of zero pixels in an image.
    """
    return torch.sum(image == 0).item()

def compare_images(image1, image2):
    """
    Compare two images by calculating the difference in the number of zero pixels.
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape")

    # Apply thresholding to image2
    image2_thresholded = threshold_image(image2)

    # Count zero pixels in each image
    num_zeros_image1 = count_zero_pixels(image1)
    num_zeros_image2 = count_zero_pixels(image2_thresholded)

    # Calculate the absolute difference in the number of zero pixels
    difference = abs(num_zeros_image1 - num_zeros_image2)

    return difference

def load_and_compare_images(path1, path2):
    """
    Load two sets of images from numpy arrays, convert them to PyTorch tensors,
    and calculate the pixel difference.
    """
    # Load numpy arrays
    image_set_1 = np.load(path1)
    image_set_2 = np.load(path2)

    # Convert to PyTorch tensors
    image_set_1 = torch.tensor(image_set_1, dtype=torch.float32)
    image_set_2 = torch.tensor(image_set_2, dtype=torch.float32)

    # Ensure that image sets are of the same shape
    if image_set_1.shape != image_set_2.shape:
        raise ValueError("Both image sets must have the same shape")

    # Compare each pair of images
    differences = [compare_images(image_set_1[i], image_set_2[i]) for i in range(image_set_1.shape[0])]

    return differences

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_dir> <epoch>")
        sys.exit(1)

    path_to_dir = sys.argv[1]
    epoch = sys.argv[2]

    path_to_first_image_set = os.path.join(path_to_dir, f'original_images_epoch{epoch}.npy')
    path_to_second_image_set = os.path.join(path_to_dir, f'reconstructed_images_epoch{epoch}.npy')

    # Calculate differences
    differences = load_and_compare_images(path_to_first_image_set, path_to_second_image_set)

    # Process the differences as needed
    # For example, printing the sum of differences for each pair
    for i, diff in enumerate(differences):
        print(f"Sum of differences for image pair {i}: {diff} pixels")

