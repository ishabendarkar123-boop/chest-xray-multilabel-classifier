"""
PyTorch Dataset class for the NIH ChestX-ray14 multi-label classification task.

Handles:
- Locating images across the 12 raw NIH folders (images_001 ... images_012)
- Standardizing image mode (L / RGBA -> RGB) to fix the inconsistency found in EDA
- Resizing + ImageNet normalization
- Train-only augmentation
- Returning (image_tensor, label_vector) pairs for the 14 disease classes
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

DISEASE_LIST = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
                 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_path_map(raw_data_dir, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    image_path_map = {}
    for folder in sorted(os.listdir(raw_data_dir)):
        if folder.startswith('images_'):
            img_subdir = os.path.join(raw_data_dir, folder, 'images')
            if os.path.isdir(img_subdir):
                for fname in os.listdir(img_subdir):
                    image_path_map[fname] = os.path.join(img_subdir, fname)

    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(image_path_map, f)

    return image_path_map


def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_path_map, split='train'):
        self.df = pd.read_csv(csv_path)
        self.image_path_map = image_path_map
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['Image Index']
        img_path = self.image_path_map[fname]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        labels = torch.tensor(row[DISEASE_LIST].values.astype('float32'))

        return image, labels