import logging

from PIL import Image

from src.models.image_enhance import Enhancer


def main():
    img = Image.open("C:/Users/Vlad/Pictures/SAM_7355.JPG")
    enhancer = Enhancer(model_name="ufpdeblur", tile_size=500, device="cuda")
    img = enhancer.enhance(img)
    img.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
