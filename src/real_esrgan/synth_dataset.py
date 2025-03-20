import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

from src.real_esrgan.image_degradation.image_process import image_degradation


class SynthDataset(Dataset):
    def __init__(self, images_paths, hr_size=256, scale_factor=2):
        super().__init__()
        self.images_paths = images_paths
        self.scale_factor = scale_factor
        self.transform = v2.RandomCrop(hr_size)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        gt = read_image(img_path)
        gt = gt.type(torch.float32) / 255.0
        gt = self.transform(gt)
        lq = image_degradation(gt, scale_factor=self.scale_factor)
        return lq.detach(), gt.detach()
