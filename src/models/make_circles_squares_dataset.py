import os 

import pandas as pd

from PIL import Image

from torch.utils.data import Dataset


class CustomDataset(Dataset):
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

        # convert shape type to binary: 0 for Square, 1 for Circle
        shape_type = 0 if self.labels.iloc[idx, 1] == "Square" else 1

        # returning image, shape type and number of shapes as a tuple
        return image, shape_type