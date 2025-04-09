from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

from src.models.image_enhance import Enhancer


def main():
    img = Image.open("C:/Users/Vlad/Pictures/SAM_7356.JPG")
    img = img.convert("RGB")
    img = ToTensor()(img).unsqueeze(0)
    enhancer = Enhancer(model_name="nafnet_realblur", tile_size=0, device="cuda")
    img = enhancer.enhance(img)
    img = to_pil_image(img.squeeze(0).clamp(0, 1))
    img.show()


if __name__ == "__main__":
    main()
