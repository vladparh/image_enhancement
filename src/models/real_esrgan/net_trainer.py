import pytorch_lightning as pl
import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class NetTrainer(pl.LightningModule):
    def __init__(self, net, lr, loss_fn):
        super().__init__()
        self.net = net
        self.lr = lr
        self.loss_fn = loss_fn
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def training_step(self, batch, batch_idx):
        lr_img, hr_img = batch
        gen_image = self.net(lr_img)
        loss = self.loss_fn(gen_image, hr_img)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr_img, hr_img = batch
        gen_img = self.net(lr_img)
        if batch_idx == 0 and type(self.logger) is pl.loggers.wandb.WandbLogger:
            images = [img for img in (gen_img[:4] * 255).type(torch.uint8).detach()]
            self.logger.log_image(key="example_images", images=images)
        val_psnr = peak_signal_noise_ratio(gen_img, hr_img)
        self.log("val_psnr", val_psnr)

    def test_step(self, batch, batch_idx):
        lr_img, hr_img = batch
        gen_img = self.net(lr_img)
        hr_img = hr_img.clamp(0, 1)
        gen_img = gen_img.clamp(0, 1)
        test_psnr = peak_signal_noise_ratio(gen_img, hr_img)
        test_ssim = structural_similarity_index_measure(
            gen_img, hr_img, data_range=(0, 1)
        )
        test_lpips = self.lpips(gen_img, hr_img)
        self.log("test_psnr", test_psnr)
        self.log("test_ssim", test_ssim)
        self.log("test_lpips", test_lpips)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
