import os
import json
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ISIC2020Dataset(Dataset):
    def __init__(self, json_path, split='training', images_dir=None, transform=None, label_map=None, image_extension='.jpg'):
        self.images_dir = images_dir
        self.transform = transform
        self.label_map = label_map
        self.image_extension = image_extension

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.samples = data[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_name = sample['image_name']
        label = sample['label']

        img_path = os.path.join(self.images_dir, image_name + self.image_extension)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            # Pass PIL image directly to torchvision transforms
            image = self.transform(image)

        if self.label_map:
            label = self.label_map[label]

        return image, label