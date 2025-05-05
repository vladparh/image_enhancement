import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.models.real_esrgan.generator import RRDBNet
from src.models.real_esrgan.net_trainer import NetTrainer
from src.models.real_esrgan.real_dataset import SuperResolutionDataset


def main():
    img_dir = "C:/Users/Vlad/Desktop/ВКР/datasets/DIV2K/valid_x4/"
    val_dataset = SuperResolutionDataset(img_dir, is_crop=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    params = {
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_block": 23,
        "num_grow_ch": 32,
        "scale": 4,
    }
    model = RRDBNet(**params)
    model.load_state_dict(
        torch.load(
            "C:/Users/Vlad/Desktop/ВКР/image_enchancement/src/models/real_esrgan/weights/RealESRGAN_x4plus.pth",
            weights_only=True,
        )["params_ema"]
    )
    module = NetTrainer(model, lr=None, loss_fn=None)
    trainer = pl.Trainer()
    trainer.test(model=module, dataloaders=val_loader)


if __name__ == "__main__":
    main()
