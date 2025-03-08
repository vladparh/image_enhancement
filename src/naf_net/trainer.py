import pytorch_lightning as pl
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio

from src.naf_net.losses.losses import PSNRLoss


class NetTrainer(pl.LightningModule):
    def __init__(self, model, lr, min_lr, n_epochs, use_split=True):
        super().__init__()
        self.model = model
        self.lr = lr
        self.min_lr = min_lr
        self.n_epochs = n_epochs
        self.use_split = use_split
        self.loss_fn = PSNRLoss()

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        blur_img, gt_img = batch
        gen_image = self.model.model(blur_img)
        loss = self.loss_fn(gen_image, gt_img)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        blur_img, gt_img = batch
        gen_img = self.model.predict(blur_img, use_split=self.use_split)
        val_metric = peak_signal_noise_ratio(gen_img, gt_img)
        self.log("val_psnr", val_metric)

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs, eta_min=self.min_lr
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
