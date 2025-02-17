import pytorch_lightning as pl
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio


class DistillationTrainer(pl.LightningModule):
    def __init__(self, teacher, student, dist_loss, target_loss, lr):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.dist_loss = dist_loss
        self.target_loss = target_loss
        self.lr = lr

    def training_step(self, batch, batch_idx):
        lr_img, hr_img = batch
        self.teacher.eval()
        with torch.no_grad():
            teacher_output = self.teacher(lr_img)
        gen_image = self.student(lr_img)
        loss = self.dist_loss(gen_image, teacher_output) + self.target_loss(
            gen_image, hr_img
        )
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr_img, hr_img = batch
        gen_img = self.student(lr_img)
        if batch_idx == 0 and type(self.logger) is pl.loggers.wandb.WandbLogger:
            images = [img for img in (gen_img[:4] * 255).type(torch.uint8).detach()]
            self.logger.log_image(key="example_images", images=images)
        val_metric = peak_signal_noise_ratio(gen_img, hr_img)
        self.log("val_psnr", val_metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
