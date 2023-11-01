import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image

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

        # Convert line type to binary: 0 for Vertical, 1 for Horizontal
        line_type = 0 if self.labels.iloc[idx, 1] == "Vertical" else 1

        return image, line_type
