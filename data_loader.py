import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import os

class CurvatureAwareAdaptiveTrainingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []  # List of image paths or numpy arrays
        self.labels = []  # Corresponding labels
        self._load_data()

    def _load_data(self):
        # Implement the logic to load data from `data_dir` and assign it to `self.data` and `self.labels`
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if isinstance(image, str):
            # Load the image from file if it's a path
            image = np.load(image)  # Example: load an image using numpy

        if self.transform:
            image = self.transform(image)

        return image, label

# Define data preprocessing and augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize the dataset and dataloader
dataset = CurvatureAwareAdaptiveTrainingDataset(data_dir='path/to/your/data', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example usage
if __name__ == "__main__":
    for images, labels in dataloader:
        print(images.shape)  # Check the shape of the loaded data
        print(labels)  # Print the corresponding labels