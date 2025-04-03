import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


def pair_crop(blur_img, gt_img, crop_size=256):
    nx = int((gt_img.shape[-2] - crop_size) * np.random.rand())
    ny = int((gt_img.shape[-1] - crop_size) * np.random.rand())
    crop_blur = v2.functional.crop(blur_img, nx, ny, crop_size, crop_size)
    crop_gt = v2.functional.crop(gt_img, nx, ny, crop_size, crop_size)
    return crop_blur, crop_gt


def pair_flip(blur_img, gt_img, p_flip):
    if np.random.rand() <= p_flip:
        flip_blur = v2.functional.horizontal_flip(blur_img)
        flip_gt = v2.functional.horizontal_flip(gt_img)
        return flip_blur, flip_gt
    else:
        return blur_img, gt_img


class TrainDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        blur_img_paths,
        gt_img_paths,
        is_crop=True,
        crop_size=256,
        is_flip=False,
        p_flip=0.5,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.blur_img_paths = blur_img_paths
        self.gt_img_paths = gt_img_paths
        self.is_crop = is_crop
        self.crop_size = crop_size
        self.is_flip = is_flip
        self.p_flip = p_flip

    def __len__(self):
        return len(self.gt_img_paths)

    def __getitem__(self, idx):
        gt_img = read_image(os.path.join(self.dataset_path, self.gt_img_paths[idx]))
        blur_img = read_image(os.path.join(self.dataset_path, self.blur_img_paths[idx]))
        gt_img = gt_img.type(torch.float32) / 255.0
        blur_img = blur_img.type(torch.float32) / 255.0
        if self.is_crop:
            blur_img, gt_img = pair_crop(blur_img, gt_img, self.crop_size)
        if self.is_flip:
            blur_img, gt_img = pair_flip(blur_img, gt_img, self.p_flip)
        return blur_img, gt_img


class TestDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        blur_img_paths,
        gt_img_paths,
        crop_size: List[int] = [500, 600],
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.blur_img_paths = blur_img_paths
        self.gt_img_paths = gt_img_paths
        self.crop_size = crop_size

    def __len__(self):
        return len(self.gt_img_paths)

    def __getitem__(self, idx):
        gt_img = read_image(os.path.join(self.dataset_path, self.gt_img_paths[idx]))
        blur_img = read_image(os.path.join(self.dataset_path, self.blur_img_paths[idx]))
        gt_img = gt_img.type(torch.float32)
        blur_img = blur_img.type(torch.float32)
        gt_img = v2.functional.center_crop(gt_img, self.crop_size)
        blur_img = v2.functional.center_crop(blur_img, self.crop_size)
        return blur_img / 255.0, gt_img / 255.0
