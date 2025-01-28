import pytorch_lightning as pl
import torch


class GANtrainer(pl.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        g_lr,
        d_lr,
        g_steps,
        gan_loss,
        percept_loss,
        l1_loss,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.g_steps = g_steps
        self.gan_loss = gan_loss
        self.percept_loss = percept_loss
        self.l1_loss = l1_loss

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        lr_img, hr_img = batch

        gen_img = self.generator(lr_img)
        fake_logits = self.discriminator(gen_img)

        if batch_idx % self.g_steps == 0:
            # train discriminator
            real_logits = self.discriminator(hr_img)
            real_loss = self.gan_loss(real_logits, target_is_real=True, is_disc=True)
            fake_loss = self.gan_loss(fake_logits, target_is_real=False, is_disc=True)
            loss = real_loss + fake_loss

            d_opt.zero_grad()
            self.manual_backward(loss)
            d_opt.step()

            self.log("d_loss", loss, prog_bar=True, on_epoch=True)
        else:
            # train generator
            percept_loss, _ = self.percept_loss(gen_img, hr_img)
            gan_loss = self.gan_loss(fake_logits, target_is_real=True, is_disc=False)
            l1_loss = self.l1_loss(gen_img, hr_img)
            loss = percept_loss + gan_loss + l1_loss

            g_opt.zero_grad()
            self.manual_backward(loss)
            g_opt.step()

            self.log("g_loss", loss, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        lr_img, hr_img = batch
        gen_img = self.generator(lr_img)
        fake_logits = self.discriminator(gen_img)
        if batch_idx == 0 and type(self.logger) is pl.loggers.wandb.WandbLogger:
            images = [img for img in (gen_img[:4] * 255).type(torch.uint8).detach()]
            self.logger.log_image(key="example_images", images=images)
        percept_loss, _ = self.percept_loss(gen_img, hr_img)
        gan_loss = self.gan_loss(fake_logits, target_is_real=True, is_disc=False)
        l1_loss = self.l1_loss(gen_img, hr_img)
        loss = percept_loss + gan_loss + l1_loss
        self.log("val_g_loss", loss)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr)
        return g_opt, d_opt
