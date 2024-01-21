from PIL import Image
from torch.utils.data import Dataset
import os
import random
import numpy as np

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        img1 = []
        if self.transform is not None:
            img1 = self.transform(img)

        return img1, label

    def __len__(self):
        return len(self.imgs)
