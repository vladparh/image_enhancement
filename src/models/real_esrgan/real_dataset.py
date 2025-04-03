import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


def pair_crop(lr_img, hr_img, hr_size=256, scale_factor=4):
    lr_size = hr_size // scale_factor
    nx = (hr_img.shape[-2] - hr_size) * np.random.rand() / hr_img.shape[-2]
    ny = (hr_img.shape[-1] - hr_size) * np.random.rand() / hr_img.shape[-1]
    crop_lr = v2.functional.crop(
        lr_img, int(nx * lr_img.shape[-2]), int(ny * lr_img.shape[-1]), lr_size, lr_size
    )
    crop_hr = v2.functional.crop(
        hr_img, int(nx * hr_img.shape[-2]), int(ny * hr_img.shape[-1]), hr_size, hr_size
    )
    return crop_lr, crop_hr


def pair_flip(lr_img, hr_img, p_flip):
    if np.random.rand() <= p_flip:
        flip_lr = v2.functional.horizontal_flip(lr_img)
        flip_hr = v2.functional.horizontal_flip(hr_img)
        return flip_lr, flip_hr
    else:
        return lr_img, hr_img


class SuperResolutionDataset(Dataset):
    def __init__(
        self,
        img_dir,
        is_crop=True,
        hr_size=256,
        scale_factor=4,
        is_flip=False,
        p_flip=0.5,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.is_crop = is_crop
        self.hr_size = hr_size
        self.scale_factor = scale_factor
        self.is_flip = is_flip
        self.p_flip = p_flip

    def __len__(self):
        return len(os.listdir(self.img_dir + "LR"))

    def __getitem__(self, idx):
        hr_img = read_image(self.img_dir + f"HR/{idx}.png")
        lr_img = read_image(self.img_dir + f"LR/{idx}.png")
        hr_img = hr_img.type(torch.float32) / 255.0
        lr_img = lr_img.type(torch.float32) / 255.0
        if self.is_crop:
            lr_img, hr_img = pair_crop(lr_img, hr_img, self.hr_size, self.scale_factor)
        if self.is_flip:
            lr_img, hr_img = pair_flip(lr_img, hr_img, self.p_flip)
        return lr_img, hr_img
