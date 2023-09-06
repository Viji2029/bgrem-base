import os
from collections import namedtuple

import cv2
import numpy as np

# import wandb
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from data import SalObjDataset
from model.u2net import U2NET_full, U2NET_lite
import PIL
from PIL import Image
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

# train_config = namedtuple('TrainConfig', 'train_base_path val_base_path batch_size epochs lr')

if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image


class U2NetLightning(pl.LightningModule):
    def __init__(self, cfg, model):
        super(U2NetLightning, self).__init__()
        self.cfg = cfg
        self.model = model

        self.bce_loss = nn.BCEWithLogitsLoss(size_average=True)
        self.mae_loss = torch.nn.L1Loss()
        #print(self.cfg["ablation"])

    def prepare_data(self):
        if self.cfg["ablation"]:
            print("Doing an ablation run")
            n_samples_train = 50
            n_samples_val= 10
            
        else:
            print("Training in progress")
            n_samples_train = None
            n_samples_val=None
            

               
        self.train_dataset = SalObjDataset(base_path=self.cfg["train_base_path"], mode="train", sz=320, rc=288,n_samples=n_samples_train)
        self.val_dataset = SalObjDataset(base_path=self.cfg["val_base_path"], mode="val", sz=320,n_samples=n_samples_val)

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        # def collate_fn(batch):
        #     imgs, masks = [list(item) for item in zip(*batch)]
        #     w = imgs[0].size()[1]
        #     h = imgs[0].size()[2]
        #     tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.float32).contiguous(
        #         memory_format=torch.channels_last)
        #     targets = torch.zeros((len(imgs), 1, h, w), dtype=torch.float32).contiguous(
        #         memory_format=torch.channels_last)
        #     for i, img in enumerate(imgs):
        #         tensor[i] += img
        #         targets[i] += masks[i]
        #     return tensor, targets

        def collate_fn(batch):
            imgs, masks = [list(item) for item in zip(*batch)]
            size = [160, 192, 224, 256, 288][np.random.randint(0, 5)]
            w = size  # imgs[0].size()[1]
            h = size  # imgs[0].size()[2]
            len_imgs = len(imgs)
            tensor = torch.zeros((len_imgs, 3, h, w), dtype=torch.float32)
            targets = torch.zeros((len_imgs, 1, h, w), dtype=torch.float32)
            for i, img in enumerate(imgs):
                img = img.unsqueeze(0)
                out = F.interpolate(img, size=(w, h))  # img
                out = out.squeeze(0)
                mask = masks[i].unsqueeze(0)
                out_m = F.interpolate(mask, size=(w, h))  # img
                out_m = out_m.squeeze(0)
                tensor[i] += out
                targets[i] += out_m
            return tensor, targets

        # train_loader = torch.utils.data.DataLoader(self.train_dataset, collate_fn=collate_fn, batch_size=self.cfg['batch_size'], shuffle=True, drop_last=True, num_workers=4)
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=8, num_workers=8, shuffle=False)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.cfg["lr"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )
        # cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 645)
        return optimizer

    def muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        loss0 = self.bce_loss(d0, labels_v)
        loss1 = self.bce_loss(d1, labels_v)
        loss2 = self.bce_loss(d2, labels_v)
        loss3 = self.bce_loss(d3, labels_v)
        loss4 = self.bce_loss(d4, labels_v)
        loss5 = self.bce_loss(d5, labels_v)
        loss6 = self.bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        return loss0, loss

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        d0, d1, d2, d3, d4, d5, d6 = self(inputs)
        loss2, loss = self.muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
        self.log("train_loss_pl", loss)
        self.logger.log_metrics({"train_loss": float(loss.cpu().detach().numpy())})
        # self.logger.log_image(key="true labels", images = list(d0))
        # wandb.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # only taking loss of final output here
        inputs, labels = batch

        d0, d1, d2, d3, d4, d5, d6 = self(inputs)
        loss2, loss = self.muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
       

        val_mae = self.mae_loss(torch.sigmoid(d0), labels)

        self.log("val_loss_pl", loss2)
        self.log("val_mae_pl", val_mae)
        self.logger.log_metrics(
            {"val_loss": float(loss2.cpu().detach().numpy()), "val_mae": float(val_mae.cpu().detach().numpy())}
        )
        self.logger.log_image(key="true val labels", images=list(labels * 255))
        self.logger.log_image("val_labels", images=list((torch.sigmoid(d0)) * 255))
        return loss2


class CheckpointEveryNSteps(pl.Callback):
    """Save a checkpoint every N steps, instead of Lightning's default that checkpoints based on
    validation loss."""

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """Check if we should save a checkpoint after every train batch."""
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


if __name__ == "__main__":

    pl.seed_everything(42)

    config = dict(
        train_base_path="/Users/dhruv/Downloads/u2net_lite_test",
        val_base_path="/Users/dhruv/Downloads/u2net_lite_test",
        batch_size=16,
        epochs=200,
        lr=0.001,
    )
    wandb_logger = WandbLogger()

    # wandb.init(config=params, project="duts_plus_sup")
    # # Config parameters are automatically set by W&B sweep agent
    # config = wandb.config

    # u2net = U2NET_full().to(memory_format=torch.channels_last)
    u2net = U2NET_full()
    # u2net = U2NET_lite()
    pl_model = U2NetLightning(config, u2net)

    # checkpoint_path = 'u2net_epoch=0125_train_loss=0.21_val_loss=0.08_val_mae=0.0171.ckpt'
    # ckpt = torch.load(checkpoint_path, map_location='cpu')
    # pl_model.load_state_dict(ckpt['state_dict'])

    train_checkpoint_train_loss = pl.callbacks.ModelCheckpoint(
        dirpath=".",
        monitor="train_loss",
        filename="u2net_train_loss_{epoch:04d}_{train_loss:.2f}",
        mode="min",
    )
    val_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=".",
        monitor="val_mae",
        filename="u2net_{epoch:04d}_{train_loss:.2f}_{val_loss:.2f}_{val_mae:.4f}",
        save_top_k=500,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config["epochs"], gpus=None, callbacks=[train_checkpoint_train_loss], process_position=2
    )
    # trainer = pl.Trainer(max_epochs=epochs, gpus=-1, callbacks=[model_checkpoint], process_position=2, precision=16, accelerator='ddp')
    # trainer = pl.Trainer(max_epochs=epochs, callbacks=[model_checkpoint], precision=16, amp_backend='apex', amp_level='O2', gpus=4)
    # trainer = pl.Trainer(
    #     max_epochs=config["epochs"],
    #     callbacks=[train_checkpoint_train_loss, val_checkpoint, CheckpointEveryNSteps(1200)],
    #     # precision=16,
    #     gpus=-1,
    #     accelerator="ddp",
    #     # check_val_every_n_epoch=10,
    #     logger=wandb_logger,
    # )

    # trainer = pl.Trainer(max_epochs=epochs, gpus=-1, deterministic=True, precision=16, callbacks=[model_checkpoint], accelerator='ddp', progress_bar_refresh_rate=100)
    # trainer = pl.Trainer(max_epochs=epochs, gpus=-1, deterministic=True, precision=16, callbacks=[model_checkpoint], accelerator='ddp', amp_backend='apex', amp_level='02')
    # trainer = pl.Trainer(max_epochs=epochs, deterministic=True, callbacks=[model_checkpoint], accelerator='ddp')

    trainer.fit(pl_model)
