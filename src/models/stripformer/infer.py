import logging

from PIL import Image

from src.models.image_enhance import Enhancer


def main():
    img = Image.open(
        "C:/Users/Vlad/Desktop/ВКР/datasets/RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene044/blur/blur_21.png"
    )
    enhancer = Enhancer(model_name="stripformer", tile_size=0, device="cuda")
    img = enhancer.enhance(img)
    img.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
