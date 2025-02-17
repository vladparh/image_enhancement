import math
import random

import numpy as np
import torch
from bestconfig import Config
from torch.nn import functional as F

from .degradations import (
    circular_lowpass_kernel,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    random_mixed_kernels,
)
from .diffjpeg import DiffJPEG
from .utils import USMSharp, filter2D

config = Config("config.yaml")


def image_degradation(img: torch.Tensor, scale_factor=4):
    # blur settings for the first degradation
    kernel_list = config["kernel_list"]
    kernel_prob = config["kernel_prob"]  # a list for each kernel probability
    blur_sigma = config["blur_sigma"]
    betag_range = config[
        "betag_range"
    ]  # betag used in generalized Gaussian blur kernels
    betap_range = config["betap_range"]  # betap used in plateau blur kernels
    sinc_prob = config["sinc_prob"]  # the probability for sinc filters

    # blur settings for the second degradation
    kernel_list2 = config["kernel_list2"]
    kernel_prob2 = config["kernel_prob2"]
    blur_sigma2 = config["blur_sigma2"]
    betag_range2 = config["betag_range2"]
    betap_range2 = config["betap_range2"]
    sinc_prob2 = config["sinc_prob2"]

    # a final sinc filter
    final_sinc_prob = config["final_sinc_prob"]

    kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
    # TODO: kernel range is now hard-coded, should be in the configure file
    pulse_tensor = torch.zeros(
        21, 21
    ).float()  # convolving with pulse tensor brings no blurry effect
    pulse_tensor[10, 10] = 1

    # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob:
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel1 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel1 = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma,
            [-math.pi, math.pi],
            betag_range,
            betap_range,
            noise_range=None,
        )
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel1 = np.pad(kernel1, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob2:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            kernel_list2,
            kernel_prob2,
            kernel_size,
            blur_sigma2,
            blur_sigma2,
            [-math.pi, math.pi],
            betag_range2,
            betap_range2,
            noise_range=None,
        )

    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------------------- the final sinc kernel ------------------------------------- #
    if np.random.uniform() < final_sinc_prob:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor

    kernel1 = torch.FloatTensor(kernel1).to(img.device)
    kernel2 = torch.FloatTensor(kernel2).to(img.device)
    sinc_kernel = sinc_kernel.to(img.device)

    usm_sharpener = USMSharp().to(img.device)
    gt = img.unsqueeze(0)
    if config["gt_usm"]:
        gt = usm_sharpener(gt)
    ori_h, ori_w = gt.size()[2:4]
    jpeger = DiffJPEG(differentiable=False).to(img.device)

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(gt, kernel1)
    # random resize
    updown_type = random.choices(["up", "down", "keep"], config["resize_prob"])[0]
    if updown_type == "up":
        scale = np.random.uniform(1, config["resize_range"][1])
    elif updown_type == "down":
        scale = np.random.uniform(config["resize_range"][0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = config["gray_noise_prob"]
    if np.random.uniform() < config["gaussian_noise_prob"]:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=config["noise_range"],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=config["poisson_scale_range"],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
        )
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*config["jpeg_range"])
    out = torch.clamp(
        out, 0, 1
    )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if np.random.uniform() < config["second_blur_prob"]:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(["up", "down", "keep"], config["resize_prob2"])[0]
    if updown_type == "up":
        scale = np.random.uniform(1, config["resize_range2"][1])
    elif updown_type == "down":
        scale = np.random.uniform(config["resize_range2"][0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(
        out,
        size=(int(ori_h / scale_factor * scale), int(ori_w / scale_factor * scale)),
        mode=mode,
    )
    # add noise
    gray_noise_prob = config["gray_noise_prob2"]
    if np.random.uniform() < config["gaussian_noise_prob2"]:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=config["noise_range2"],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=config["poisson_scale_range2"],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
        )

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out, size=(ori_h // scale_factor, ori_w // scale_factor), mode=mode
        )
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*config["jpeg_range2"])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*config["jpeg_range2"])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out, size=(ori_h // scale_factor, ori_w // scale_factor), mode=mode
        )
        out = filter2D(out, sinc_kernel)

    # clamp and round
    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    return lq.squeeze()
