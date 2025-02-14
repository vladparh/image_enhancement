import pytorch_lightning as pl
import torch
from discriminator import UNetDiscriminatorSN
from gan_trainer import GANtrainer
from generator import RRDBNet
from logger import get_logger
from losses.basic_loss import L1Loss, PerceptualLoss
from losses.gan_loss import GANLoss
from real_dataset import SuperResolutionDataset
from torch.utils.data import DataLoader


def main():
    torch.set_float32_matmul_precision("medium")
    train_dataset = SuperResolutionDataset(
        img_dir="C:/Users/Vlad/Desktop/ВКР/sr_dataset/train/"
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    generator = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32
    )
    generator.load_state_dict(torch.load("weights/RealESRGAN_x4plus.pth")["params_ema"])

    discriminator = UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True)
    discriminator.load_state_dict(
        torch.load("weights/RealESRGAN_x4plus_netD.pth")["params"]
    )

    layer_weigts = {
        "conv1_2": 0.1,
        "conv2_2": 0.1,
        "conv3_4": 1,
        "conv4_4": 1,
        "conv5_4": 1,
    }
    percept_loss = PerceptualLoss(
        layer_weights=layer_weigts,
        vgg_type="vgg19",
        use_input_norm=True,
        range_norm=False,
        perceptual_weight=1.0,
        style_weight=0,
        criterion="l1",
    )

    gan_loss = GANLoss(
        gan_type="vanilla", real_label_val=1.0, fake_label_val=0.0, loss_weight=1e-1
    )

    l1_loss = L1Loss(loss_weight=1.0, reduction="mean")

    module = GANtrainer(
        generator=generator,
        discriminator=discriminator,
        g_steps=2,
        g_lr=1e-4,
        d_lr=1e-4,
        percept_loss=percept_loss,
        gan_loss=gan_loss,
        l1_loss=l1_loss,
    )

    logger_conf = {
        "label": "mlflow",
        "experiment_name": "real-esrgan train",
        "run_name": "first run",
        "mlflow_save_dir": ".",
        "tracking_uri": "http://127.0.0.1:5000",
    }

    logger = get_logger(logger_conf)

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath="weights/", monitor="g_loss"
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        logger=logger,
        callbacks=[model_checkpoint],
    )

    trainer.fit(module, train_loader)


if __name__ == "__main__":
    main()
