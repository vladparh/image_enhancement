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
    img = Image.open("C:/Users/Vlad/Pictures/SAM_1700.JPG")
    enhancer = Enhancer(model_name="mlwnet", tile_size=0, device="cuda")
    img = enhancer.enhance(img)
    img.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
