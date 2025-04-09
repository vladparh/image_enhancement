import logging

from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

from src.models.image_enhance import Enhancer


def main():
    logging.basicConfig(level=logging.INFO)
    img = Image.open("C:/Users/Vlad/Downloads/photo_5452073207310971245_y.jpg")
    img = img.convert("RGB")
    img = ToTensor()(img).unsqueeze(0)
    enhancer = Enhancer(model_name="real_esrgan_x4", tile_size=500, device="cuda")
    img = enhancer.enhance(img)
    img = to_pil_image(img.squeeze(0).clamp(0, 1))
    img.show()


if __name__ == "__main__":
    main()
