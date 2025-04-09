import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
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


def pair_resize(blur_img, gt_img, scale=(0.8, 1.2)):
    scale_factor = np.random.uniform(*scale)
    print(scale_factor)
    resize_blur = F.interpolate(
        blur_img.unsqueeze(0), scale_factor=scale_factor, mode="bicubic"
    )
    resize_gt = F.interpolate(
        gt_img.unsqueeze(0), scale_factor=scale_factor, mode="bicubic"
    )
    return resize_blur.squeeze(), resize_gt.squeeze()


class TrainDataset(Dataset):
    """
    Train dataset

    Attributes:
        dataset_path (str): path to dataset
        blur_img_paths (List[str]): paths to blur images
        gt_img_paths (List[str]): paths to blur images
        is_crop (bool): use or not crop
        crop_size (int): crop size
        is_flip (bool): use flip or not
        p_flip (float): probability of flip
        is_resize (bool): use resize or not
        scale (Tuple[float]): range for resize
    """

    def __init__(
        self,
        dataset_path,
        blur_img_paths,
        gt_img_paths,
        is_crop=True,
        crop_size=256,
        is_flip=False,
        p_flip=0.5,
        is_resize=False,
        scale=(0.8, 1.2),
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.blur_img_paths = blur_img_paths
        self.gt_img_paths = gt_img_paths
        self.is_crop = is_crop
        self.crop_size = crop_size
        self.is_flip = is_flip
        self.p_flip = p_flip
        self.is_resize = is_resize
        self.scale = scale

    def __len__(self):
        return len(self.gt_img_paths)

    def __getitem__(self, idx):
        gt_img = read_image(os.path.join(self.dataset_path, self.gt_img_paths[idx]))
        blur_img = read_image(os.path.join(self.dataset_path, self.blur_img_paths[idx]))
        gt_img = gt_img.type(torch.float32) / 255.0
        blur_img = blur_img.type(torch.float32) / 255.0
        if self.is_resize:
            blur_img, gt_img = pair_resize(blur_img, gt_img, self.scale)
        if self.is_crop:
            blur_img, gt_img = pair_crop(blur_img, gt_img, self.crop_size)
        if self.is_flip:
            blur_img, gt_img = pair_flip(blur_img, gt_img, self.p_flip)
        return blur_img.clamp(0, 1), gt_img.clamp(0, 1)


class TestDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        blur_img_paths,
        gt_img_paths,
        is_crop=True,
        crop_size: List[int] = [500, 600],
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.blur_img_paths = blur_img_paths
        self.gt_img_paths = gt_img_paths
        self.is_crop = is_crop
        self.crop_size = crop_size

    def __len__(self):
        return len(self.gt_img_paths)

    def __getitem__(self, idx):
        gt_img = read_image(os.path.join(self.dataset_path, self.gt_img_paths[idx]))
        blur_img = read_image(os.path.join(self.dataset_path, self.blur_img_paths[idx]))
        gt_img = gt_img.type(torch.float32)
        blur_img = blur_img.type(torch.float32)
        if self.is_crop:
            gt_img = v2.functional.center_crop(gt_img, self.crop_size)
            blur_img = v2.functional.center_crop(blur_img, self.crop_size)
        return blur_img / 255.0, gt_img / 255.0
