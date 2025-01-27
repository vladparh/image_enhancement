import os

import torch
from generator import RRDBNet
from torchvision.io import read_image
from torchvision.utils import save_image


def inference(in_dir, out_dir, original=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32
    )
    model.to(device)
    if original:
        model.load_state_dict(torch.load("weights/RealESRGAN_x4plus.pth")["params_ema"])
    else:
        checkpoint = torch.load("weights/epoch=9-step=1000.ckpt")
        model_weights = checkpoint["state_dict"]
        generator_model_weights = {}
        for key in list(model_weights):
            if "generator" in key:
                generator_model_weights[
                    key.replace("generator.", "")
                ] = model_weights.pop(key)
        model.load_state_dict(generator_model_weights)
    model.eval()

    for file in os.listdir(in_dir):
        img = read_image(in_dir + file)
        img = img.to(device)
        img = img.unsqueeze(0)
        img = img.type(torch.float32) / 255.0
        upscale_img = model(img)
        if original:
            prefix = "_upscale_origin.png"
        else:
            prefix = "_upscale.png"
        new_filename = out_dir + os.path.splitext(file)[0] + prefix
        save_image(upscale_img, new_filename)


if __name__ == "__main__":
    inference("input/", "output/")
