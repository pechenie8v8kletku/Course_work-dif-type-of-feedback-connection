import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np



'''
обычный класс датасета для загрузки изображений в лоадер

'''
class Dog_Set(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for i, label_name in enumerate(sorted(os.listdir(root_dir))):
            full = os.path.join(root_dir, label_name)
            if not os.path.isdir(full):
                continue
            label = i
            for fname in os.listdir(full):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(full, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            augmented = self.transform(image=np.array(img))
            img = augmented["image"]


        return img, torch.tensor(label, dtype=torch.long)