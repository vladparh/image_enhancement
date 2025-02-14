import matplotlib.pyplot as plt
import torch
from dataset import SuperResolutionDataset
from image_degradation.image_process import image_degradation


def display(img1, img2):
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax1.imshow(img1)
    ax2.imshow(img2)


sr_dataset = SuperResolutionDataset("C:/Users/Vlad/Desktop/ВКР/sr_dataset/train/")
lr_img, hr_img = sr_dataset[1]
lq = image_degradation(hr_img)
hr_img_1 = (hr_img * 255).type(torch.uint8)
lq_1 = (lq * 255).type(torch.uint8)
display(lq_1.permute((1, 2, 0)).numpy(), hr_img_1.permute((1, 2, 0)).numpy())
