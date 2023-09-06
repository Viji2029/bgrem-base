# import wandb
import argparse
import glob
import os
from datetime import datetime

import mlflow
import pytorch_lightning as pl
from dotenv import load_dotenv
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
from train_lightning import CheckpointEveryNSteps, U2NET_full, U2NetLightning

load_dotenv(".env")

# def parse_args():


def train(args):
    print("Args: ", args)

    print("Train: ", glob.glob(args.train_base_path + "/*"))
    print("Val: ", glob.glob(args.val_base_path + "/*"))
    print("Ablation : ", args.ablation)

    config = vars(args)

    pl.seed_everything(args.seed)

    curr_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    wandb_logger = WandbLogger(project="BG_REM",name="expt_1",log_model="all")
    mlflow.pytorch.autolog()
    mlflow_logger = MLFlowLogger(experiment_name="u2net", run_name=os.getenv("RUN_NAME") or curr_datetime)

    # wandb.init(config=params, project="duts_plus_sup")
    # # Config parameters are automatically set by W&B sweep agent
    # config = wandb.config

    # u2net = U2NET_full().to(memory_format=torch.channels_last)
    u2net = U2NET_full()
    # u2net = U2NET_lite()
    print("Call Lightning")

    pl_model = U2NetLightning(config, u2net)

    # checkpoint_path = 'u2net_epoch=0125_train_loss=0.21_val_loss=0.08_val_mae=0.0171.ckpt'
    # ckpt = torch.load(checkpoint_path, map_location='cpu')
    # pl_model.load_state_dict(ckpt['state_dict'])

    ckpt_dir = f"{args.out_path}/checkpoints/{curr_datetime}"
    train_checkpoint_train_loss = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="train_loss_pl",
        filename="u2net_train_loss_{epoch:04d}_{train_loss:.2f}",
        mode="min",
    )
    val_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_mae_pl",
        filename="u2net_{epoch:04d}_{train_loss:.2f}_{val_loss_pl:.2f}_{val_mae_pl:.4f}",
        save_top_k=500,
        mode="min",
    )

    # trainer = pl.Trainer(
    #     max_epochs=config["epochs"],
    #     gpus=None,
    #     callbacks=[train_checkpoint_train_loss],
    #     process_position=2,
    # )
    # trainer = pl.Trainer(max_epochs=epochs, gpus=-1, callbacks=[model_checkpoint], process_position=2, precision=16, accelerator='ddp')
    # trainer = pl.Trainer(max_epochs=epochs, callbacks=[model_checkpoint], precision=16, amp_backend='apex', amp_level='O2', gpus=4)
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        callbacks=[
            train_checkpoint_train_loss,
            val_checkpoint,
            # CheckpointEveryNSteps(1200),
        ],
        # precision=16,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        logger=[wandb_logger, mlflow_logger]
        # check_val_every_n_epoch=10,
    )

    # trainer = pl.Trainer(max_epochs=epochs, gpus=-1, deterministic=True, precision=16, callbacks=[model_checkpoint], accelerator='ddp', progress_bar_refresh_rate=100)
    # trainer = pl.Trainer(max_epochs=epochs, gpus=-1, deterministic=True, precision=16, callbacks=[model_checkpoint], accelerator='ddp', amp_backend='apex', amp_level='02')
    # trainer = pl.Trainer(max_epochs=epochs, deterministic=True, callbacks=[model_checkpoint], accelerator='ddp')

    trainer.fit(pl_model)


if __name__ == "__main__":
    train()
