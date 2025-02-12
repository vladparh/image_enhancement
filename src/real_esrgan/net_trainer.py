import pytorch_lightning as pl
import torch


class NetTrainer(pl.LightningModule):
    def __init__(self, net, lr, loss_fn):
        super().__init__()
        self.net = net
        self.lr = lr
        self.loss_fn = loss_fn

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
        loss = self.loss_fn(gen_img, hr_img)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
