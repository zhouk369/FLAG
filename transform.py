from PIL import Image
import numpy as np

import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import torch
import random


class TwoTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        v1 = self.base_transform(x)
        v2 = self.base_transform(x)
        return [v1, v2]


class OneOfTrans:
    """random select one of from the input transform list"""

    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, x):
        return self.base_transforms[random.randint(0, len(self.base_transforms) - 1)](x)


class ALBU_AUG:
    def __init__(self, base_transform):
        self.transform = base_transform

    def __call__(self, x):
        if isinstance(x, Image.Image):
            x = np.asarray(x)
        return self.transform(image=x)['image']


def get_augs(name="base", norm="imagenet", size=299):
    IMG_SIZE = size
    if norm == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif norm == "0.5":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]

    if name == "None":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif name == "Valid":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif name == "Test":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
