import torch
from generator import RRDBNet
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from upscale_image import Enchacer


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img = Image.open("C:/Users/Vlad/Pictures/Screenshots/SAM_7310.JPG")
    img = img.convert("RGB")
    img = ToTensor()(img).unsqueeze(0)
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )
    model.load_state_dict(
        torch.load(
            "C:/Users/Vlad/Desktop/ВКР/image_enchancement/src/real_esrgan/weights/RealESRGAN_x4plus.pth"
        )["params_ema"]
    )
    # checkpoint = torch.load("C:/Users/Vlad/Desktop/ВКР/image_enchancement/src/real_esrgan/weights/finetune_model_23B_x2_val_psnr=27.77.ckpt")
    # model_weights = checkpoint["state_dict"]
    # generator_model_weights = {}
    # for key in list(model_weights):
    #     if "generator" in key:
    #         generator_model_weights[
    #             key.replace("generator.", "")
    #         ] = model_weights.pop(key)
    # model.load_state_dict(generator_model_weights)
    enhancer = Enchacer(scale=4, model=model, tile_size=400, device=device)
    img = enhancer.enhance(img)
    img = to_pil_image(img.squeeze(0).clamp(0, 1))
    img.show()


if __name__ == "__main__":
    main()
