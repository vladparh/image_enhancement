import logging

from PIL import Image

from src.models.image_enhance import Enhancer


def main():
    img = Image.open("C:/Users/Vlad/Pictures/SAM_1627.JPG")
    enhancer = Enhancer(model_name="mlwnet", tile_size=1000, device="cuda")
    img = enhancer.enhance(img)
    img.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
