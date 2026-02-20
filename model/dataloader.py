import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    """
    Custom dataset that reads image paths and labels from a text file.
    """
    def __init__(self, txt_file, data_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = []
        self.data_dir = data_dir
        self.transform = transform
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                img_path = line[0]
                label = int(line[1])
                self.img_labels.append((img_path, label))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.data_dir, self.img_labels[idx][0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx][1]

        if self.transform:
            image = self.transform(image)

        return image, label
