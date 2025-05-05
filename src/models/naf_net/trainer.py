import pytorch_lightning as pl
import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


class NetTrainer(pl.LightningModule):
    def __init__(self, model, lr, min_lr, n_epochs, loss_fn, use_scheduler=True):
        super().__init__()
        self.model = model
        self.lr = lr
        self.min_lr = min_lr
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.use_scheduler = use_scheduler

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        blur_img, gt_img = batch
        gen_image = self.model(blur_img)
        loss = self.loss_fn(gen_image, gt_img)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        blur_img, gt_img = batch
        gen_img = self.model(blur_img)
        val_psnr = peak_signal_noise_ratio(gen_img, gt_img)
        val_ssim = structural_similarity_index_measure(
            gen_img, gt_img, data_range=(0, 1)
        )
        self.log("val_psnr", val_psnr)
        self.log("val_ssim", val_ssim)

    def on_train_epoch_end(self):
        if self.use_scheduler:
            sch = self.lr_schedulers()
            sch.step()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), betas=(0.9, 0.9), weight_decay=0, lr=self.lr
        )
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.n_epochs, eta_min=self.min_lr
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
