import logging

from PIL import Image

from src.models.image_enhance import Enhancer


def main():
    logging.basicConfig(level=logging.INFO)
    img = Image.open(
        "C:/Users/Vlad/Pictures/Screenshots/0878ce1ef19d6369d809c7a4828eb2cb--porsche-bombshell.jpg"
    )
    enhancer = Enhancer(model_name="real_esrgan_x4", tile_size=500, device="cuda")
    img = enhancer.enhance(img)
    img.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
