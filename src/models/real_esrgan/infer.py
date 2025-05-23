import logging
import time

from PIL import Image

from src.models.image_enhance import Enhancer


def timeit(func):
    """
    Decorator for measuring function's running time.
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print(
            "Processing time of %s(): %.2f seconds."
            % (func.__qualname__, time.time() - start_time)
        )
        return result

    return measure_time


@timeit
def main():
    img = Image.open("C:/Users/Vlad/Downloads/camel.jpeg")
    enhancer = Enhancer(model_name="real_esrgan_x4", tile_size=256, device="cuda")
    img = enhancer.enhance(img)
    img.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
