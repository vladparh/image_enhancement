import logging
import math

import torch
from bestconfig import Config
from torch.nn import functional as F

from src.models.naf_net.model.NAFNet_arch import NAFNetLocal
from src.models.real_esrgan.generator import RRDBNet


class Enhancer:
    """
    Load model and enhance image

    Attributes:
        model_name (str): real_esrgan_x2, real_esrgan_x4, nafnet_realblur or nafnet_sidd
        tile_size (int): tile size for image splitting
        tile_pad (int): pad size for tile
        pre_pad (int): pad size for image
        device (str): device
    """

    def __init__(
        self,
        model_name: str,
        tile_size: int = 400,
        tile_pad: str = 10,
        pre_pad: int = 10,
        device: str = None,
    ):
        self.model_name = model_name
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.scale = None
        self.mod_scale = None
        self.model = None

    def load_model(self):
        config = Config("model_configs.yaml")
        if self.model_name == "real_esrgan_x2":
            self.scale = config[self.model_name]["scale"]
            self.model = RRDBNet(**config[self.model_name]["params"])
            self.model.load_state_dict(
                torch.load(config[self.model_name]["weights_path"], weights_only=True)[
                    "params_ema"
                ]
            )
            self.model = self.model.to(self.device)
        elif self.model_name == "real_esrgan_x4":
            self.scale = config[self.model_name]["scale"]
            self.model = RRDBNet(**config[self.model_name]["params"])
            self.model.load_state_dict(
                torch.load(config[self.model_name]["weights_path"], weights_only=True)[
                    "params_ema"
                ]
            )
            self.model = self.model.to(self.device)
        elif self.model_name == "nafnet_realblur":
            self.scale = config[self.model_name]["scale"]
            self.model = NAFNetLocal(**config[self.model_name]["params"])
            checkpoint = torch.load(
                config[self.model_name]["weights_path"],
                weights_only=True,
            )
            model_weights = checkpoint["state_dict"]
            generator_model_weights = {}
            for key in list(model_weights):
                if "model" in key:
                    generator_model_weights[
                        key.replace("model.", "")
                    ] = model_weights.pop(key)
            self.model.load_state_dict(generator_model_weights)
            self.model = self.model.to(self.device)
        elif self.model_name == "nafnet_sidd":
            self.scale = config[self.model_name]["scale"]
            self.model = NAFNetLocal(**config[self.model_name]["params"])
            self.model.load_state_dict(
                torch.load(config[self.model_name]["weights_path"], weights_only=True)[
                    "params"
                ]
            )
            self.model = self.model.to(self.device)
        self.model.eval()

    def pre_process(self, img):
        """
        Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        self.img = img
        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if h % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - h % self.mod_scale
            if w % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - w % self.mod_scale
            self.img = F.pad(
                self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect"
            )

    def process(self):
        # model inference
        self.output = self.model(self.img.to(self.device)).cpu()

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile.to(self.device)).cpu()
                except RuntimeError as error:
                    logging.error(error)
                logging.info(f"\tTile {tile_idx}/{tiles_x * tiles_y}")

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.mod_pad_h * self.scale,
                0 : w - self.mod_pad_w * self.scale,
            ]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.pre_pad * self.scale,
                0 : w - self.pre_pad * self.scale,
            ]
        return self.output

    @torch.no_grad()
    def enhance(self, img: torch.Tensor) -> torch.Tensor:
        try:
            self.load_model()
        except Exception:
            logging.error("LoadModelError", exc_info=True)
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process()
        return output_img
