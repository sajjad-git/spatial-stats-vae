#%%
import os
import csv
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

save_dir = "../../data/three_to_five_lines_diag"
image_save_dir = os.path.join(save_dir, 'images')
labels_save_dir = os.path.join(save_dir, "labels.csv")
num_images = 10000
img_size = 224
min_num_lines = 3
max_num_lines = 5
def is_overlapping(line1, line2):
    def is_vertical(line):
        return line[0] == line[2]

    def is_horizontal(line):
        return line[1] == line[3]

    # Check for vertical lines overlap
    if is_vertical(line1):
        if is_vertical(line2):
            return abs(line1[0] - line2[0]) < 10 and not (line1[3] < line2[1] or line2[3] < line1[1])
        return False  # No overlap between vertical and non-vertical lines

    # Check for horizontal lines overlap
    if is_horizontal(line1):
        if is_horizontal(line2):
            return abs(line1[1] - line2[1]) < 10 and not (line1[2] < line2[0] or line2[2] < line1[0])
        return False  # No overlap between horizontal and non-horizontal lines

    # For diagonal lines, use the existing diagonal overlap check
    return check_diagonal_overlap(line1, line2)

def check_diagonal_overlap(line1, line2):
    def get_aabb(line):
        x_coords, y_coords = [line[0], line[2]], [line[1], line[3]]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def aabb_overlap(aabb1, aabb2):
        return not (aabb1[2] < aabb2[0] or aabb1[0] > aabb2[2] or aabb1[3] < aabb2[1] or aabb1[1] > aabb2[3])

    aabb1 = get_aabb(line1)
    aabb2 = get_aabb(line2)

    return aabb_overlap(aabb1, aabb2)

def generate_images():
    
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    with open(labels_save_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImageName", "LineType", "NumberOfLines"])

        def draw_line(img_size, existing_lines, draw_func):
            max_attempts = 50
            for _ in range(max_attempts):
                if draw_func(img_size, existing_lines):
                    return

        def random_vertical_line(img_size, existing_lines):
            draw = ImageDraw.Draw(img)
            line_length = np.random.randint(img_size//2, img_size)
            start_point = (np.random.randint(0, img_size), np.random.randint(0, img_size - line_length))
            end_point = (start_point[0], start_point[1] + line_length)
            line = (*start_point, *end_point)

            for existing_line in existing_lines:
                if is_overlapping(line, existing_line):
                    return False
            
            line_thickness = np.random.randint(5, 10)
            draw.line([start_point, end_point], fill=0, width=line_thickness)
            existing_lines.append(line)
            return True

        def random_horizontal_line(img_size, existing_lines):
            draw = ImageDraw.Draw(img)
            line_length = np.random.randint(img_size//2, img_size)
            start_point = (np.random.randint(0, img_size - line_length), np.random.randint(0, img_size))
            end_point = (start_point[0] + line_length, start_point[1])
            line = (*start_point, *end_point)

            for existing_line in existing_lines:
                if is_overlapping(line, existing_line):
                    return False

            line_thickness = np.random.randint(5, 10)
            draw.line([start_point, end_point], fill=0, width=line_thickness)
            existing_lines.append(line)
            return True
        def random_ne_sw_diagonal_line(img_size, existing_lines):
            draw = ImageDraw.Draw(img)
            fixed_slope = -1  # Negative slope for NE-SW

            # Random point within image boundaries
            random_point = (np.random.randint(img_size), np.random.randint(img_size))

            # Determine line length and calculate end points
            line_length = np.random.randint(img_size//4, img_size//2)
            half_length = line_length // 2

            start_x = random_point[0] - half_length
            end_x = random_point[0] + half_length
            start_y = random_point[1] - fixed_slope * half_length
            end_y = random_point[1] + fixed_slope * half_length

            # Ensure the line is within the image boundaries
            if not (0 <= start_x < img_size and 0 <= end_x < img_size and
                    0 <= start_y < img_size and 0 <= end_y < img_size):
                return False

            line = (start_x, start_y, end_x, end_y)

            for existing_line in existing_lines:
                if is_overlapping(line, existing_line):
                    return False

            line_thickness = np.random.randint(5, 10)
            draw.line([start_x, start_y, end_x, end_y], fill=0, width=line_thickness)
            existing_lines.append(line)
            return True


        def random_nw_se_diagonal_line(img_size, existing_lines):
            draw = ImageDraw.Draw(img)
            fixed_slope = 1  # Positive slope for NW-SE

            # Random point within image boundaries
            random_point = (np.random.randint(img_size), np.random.randint(img_size))

            # Determine line length and calculate end points
            line_length = np.random.randint(img_size//4, img_size//2)
            half_length = line_length // 2

            start_x = random_point[0] - half_length
            end_x = random_point[0] + half_length
            start_y = random_point[1] - fixed_slope * half_length
            end_y = random_point[1] + fixed_slope * half_length

            # Ensure the line is within the image boundaries
            if not (0 <= start_x < img_size and 0 <= end_x < img_size and
                    0 <= start_y < img_size and 0 <= end_y < img_size):
                return False

            line = (start_x, start_y, end_x, end_y)

            for existing_line in existing_lines:
                if is_overlapping(line, existing_line):
                    return False

            line_thickness = np.random.randint(5, 10)
            draw.line([start_x, start_y, end_x, end_y], fill=0, width=line_thickness)
            existing_lines.append(line)
            return True


        for i in tqdm(range(num_images)):
            img = Image.new('1', (img_size, img_size), color=1)
            existing_lines = []
            line_type = np.random.randint(4)  # Now 0 to 3 for four line types
            if min_num_lines == max_num_lines:
                num_lines = max_num_lines
            else:
                num_lines = np.random.randint(min_num_lines, max_num_lines)
            
            for _ in range(num_lines):
                if line_type == 0:
                    draw_line(img_size, existing_lines, random_vertical_line)
                elif line_type == 1:
                    draw_line(img_size, existing_lines, random_horizontal_line)
                elif line_type == 2:
                    draw_line(img_size, existing_lines, random_ne_sw_diagonal_line)
                else:  # line_type == 3
                    draw_line(img_size, existing_lines, random_nw_se_diagonal_line)

            img_name = f'img_{i}.png'
            img.save(os.path.join(image_save_dir, img_name))
             # Label lines as Vertical (0), Horizontal (1), NE-SW (2), NW-SE (3)
            if line_type == 0:
                label = "Vertical"
            elif line_type == 1:
                label = "Horizontal"
            elif line_type == 2:
                label = "NE-SW"
            else:  # line_type == 3
                label = "NW-SE"

            writer.writerow([img_name, label, num_lines])

generate_images()

#%%
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import torch
import torchvision

class LinesDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        image = Image.open(os.path.join(self.img_dir, img_name))

        if self.transform:
            image = self.transform(image)

        line_label = self.labels.iloc[idx, 1]
        if line_label == "Vertical":
            line_type = 0
        elif line_label == "Horizontal":
            line_type = 1
        elif line_label == "NE-SW":
            line_type = 2
        else:  # "NW-SE"
            line_type = 3

        return image, line_type

# Transforming the PIL Image to tensors
transformations = transforms.Compose([
    transforms.ToTensor(),
])

# Initialize your Dataset
dataset = LinesDataset(labels_save_dir, image_save_dir, transformations)

# Initialize DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

# %%
import matplotlib.pyplot as plt


# %%
for i in range(50):
    j = np.random.randint(0, 10000)
    test = dataset[j]
    print(test[1])
    plt.figure()
    plt.imshow(test[0][0])
# %%