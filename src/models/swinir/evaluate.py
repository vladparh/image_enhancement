import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.models.real_esrgan.net_trainer import NetTrainer
from src.models.real_esrgan.real_dataset import SuperResolutionDataset
from src.models.swinir.model import SwinIR


def main():
    img_dir = "C:/Users/Vlad/Desktop/ВКР/datasets/DIV2K/valid_x4/"
    val_dataset = SuperResolutionDataset(img_dir, is_crop=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    params = {
        "upscale": 4,
        "in_chans": 3,
        "img_size": 64,
        "window_size": 8,
        "img_range": 1.0,
        "depths": [6, 6, 6, 6, 6, 6],
        "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6],
        "mlp_ratio": 2,
        "upsampler": "nearest+conv",
        "resi_connection": "1conv",
    }
    model = SwinIR(**params)
    model.load_state_dict(
        torch.load(
            "C:/Users/Vlad/Desktop/ВКР/image_enchancement/src/models/swinir/weights/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
            weights_only=True,
        )["params_ema"]
    )
    module = NetTrainer(model, lr=None, loss_fn=None)
    trainer = pl.Trainer()
    trainer.test(model=module, dataloaders=val_loader)


if __name__ == "__main__":
    main()
