import os 

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset

class DiagonalLinesDataset(Dataset):
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
