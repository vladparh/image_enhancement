import torch
from generator import RRDBNet
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from upscale_image import Enchacer


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img = Image.open(
        "C:/Users/Vlad/Pictures/Screenshots/Снимок экрана 2024-05-17 150213.png"
    )
    img = img.convert("RGB")
    img = ToTensor()(img).unsqueeze(0)
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
    )
    model.load_state_dict(
        torch.load(
            "C:/Users/Vlad/Desktop/ВКР/image_enchancement/src/real_esrgan/weights/RealESRGAN_x2plus.pth"
        )["params_ema"]
    )
    enhancer = Enchacer(scale=2, model=model, tile_size=400, device=device)
    img = enhancer.enhance(img)
    img = to_pil_image(img.squeeze(0).clamp(0, 1))
    img.show()


if __name__ == "__main__":
    main()
