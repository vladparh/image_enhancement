import torch
from model.NAFNet_arch import NAFNetLocal
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

from src.real_esrgan.upscale_image import Enchacer


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # img = Image.open(
    #     "C:/Users/Vlad/Desktop/ВКР/datasets/RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene230/blur/blur_2.png"
    # )
    img = Image.open("C:/Users/Vlad/Pictures/SAM_7356.JPG")
    img = img.convert("RGB")
    img = ToTensor()(img).unsqueeze(0)
    model = NAFNetLocal(
        width=32,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    )
    # model.load_state_dict(
    #    torch.load(
    #        "C:/Users/Vlad/Desktop/ВКР/image_enchancement/src/naf_net/weights/NAFNet-GoPro-width32.pth"
    #    )["params"]
    # )
    checkpoint = torch.load(
        "C:/Users/Vlad/Desktop/ВКР/image_enchancement/src/naf_net/weights/last_50_epochs.ckpt"
    )
    model_weights = checkpoint["state_dict"]
    generator_model_weights = {}
    for key in list(model_weights):
        if "model" in key:
            generator_model_weights[key.replace("model.", "")] = model_weights.pop(key)
    model.load_state_dict(generator_model_weights)
    model = model.to(device)
    enhancer = Enchacer(scale=1, model=model, tile_size=0, device=device, pre_pad=0)
    img = enhancer.enhance(img)
    img = to_pil_image(img.squeeze(0).clamp(0, 1))
    img.show()


if __name__ == "__main__":
    main()
