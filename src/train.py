import argparse
import yaml
import os
import shutil

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_info
from transformers import VideoMAEImageProcessor

from asl_dataset import RGBDSkel_Dataset
from lightning_asl import SignClassificationLightning


def train(config_f):
    config = read_config(config_f)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("WORLD_SIZE:", world_size)

    L.seed_everything(config["seed"], workers=True)

    # save dir
    save_dir = config["save"]
    if int(os.environ.get("RANK", 0)) == 0:
        if os.path.eixsts(save_dir):
            rank_zero_info(f"deleting {save_dir}")
            shutil.rmtree(save_dir)
        rank_zero_info(f"creating {save_dir}")
        os.mkdir(save_dir)

        checkpoints_dir = os.path.join(save_dir, "checkpoints")
        logs_dir = os.path.join(save_dir, "logs")
        tb_dir = os.path.join(save_dir, "tb")

        os.mkdir(checkpoints_dir)
        os.mkdir(logs_dir)
        os.mkdir(tb_dir)

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    modalities = tuple(config["modalities"])

    # load train dataloader
    train_dataset = RGBDSkel_Dataset(
        annotations=config["train_csv"],
        processor=processor,
        num_frames=config["num_frames"],
        modalities=modalities
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # load val dataloader
    val_dataset = RGBDSkel_Dataset(
        annotations=config["val_csv"],
        processor=processor,
        num_frames=config["num_frames"],
        modalities=modalities
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # load model
    lightning_model = SignClassificationLightning(config=config)

    # callbacks
    early_stopping = EarlyStopping(monitor="val_acc", mode="max", patience=config["early_stop"], verbose=True)
    top_k_model_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="{epoch}-{step}-val_loss_{val_loss:.4f}-val_acc_{val_acc:.4f}",
        save_top_k=config["save_top_k"],
        monitor="val_acc",
        mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    print_callback = PrintCallback()
    train_callbacks = [
        early_stopping,
        top_k_model_checkpoint,
        lr_monitor,
        print_callback
    ]
    logger = CSVLogger(save_dir=logs_dir)
    tb_logger = TensorBoardLogger(save_dir=tb_dir)

    # set parallelization
    if config["n_gpus"] >= 1:
        strategy = "ddp"
    else:
        strategy = "auto"

    # get trainer
    trainer = L.Trainer(
        max_steps=config["max_steps"],
        val_check_interval=config["val_interval"],
        accelerator=config["device"],
        default_root_dir=save_dir,
        callbacks=train_callbacks,
        logger=[logger, tb_logger],
        deterministic=True,
        strategy=strategy
        # gradient_clip_val=config["gradient_clip_val"]
    )

    # train :)
    trainer.fit(
        lightning_model,
        train_dataloader,
        val_dataloader
    )


def test(config_f):
    config = read_config(config_f)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("WORLD_SIZE:", world_size)

    L.seed_everything(config["seed"], workers=True)

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    modalities = tuple(config["modalities"])

    # load test dataloader
    test_dataset = RGBDSkel_Dataset(
        annotations=config["test_csv"],
        processor=processor,
        num_frames=config["num_frames"],
        modalities=modalities
    )
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        rank_zero_info("################# (Lightning) #################")
        rank_zero_info("######          TRAINING STARTED         ######")
        rank_zero_info("###############################################")
    def on_train_end(self, trainer, pl_module):
        rank_zero_info("################# (Lightning) #################")
        rank_zero_info("#######          TRAINING ENDED         #######")
        rank_zero_info("###############################################")


def read_config(f):
    with open(f) as inf:
        config = yaml.safe_load(inf)
    rank_zero_info(f"CONFIG: {csv_f}")
    for k, v in config.items():
        rank_zero_info(f"\t-t{k}=`{v}`, {type(v)}")
    rank_zero_info("\n\n")
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-m", "--mode", choices = ["TRAIN", "TEST"])
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t-{k}=`{v}`")
    print("\n\n")
    return args

if __name__ == "__main__":
    print("############")
    print("# train.py #")
    print("############")
    args = get_args()
    if args.mode == "TRAIN":
        print("- TRAINING -")
        train(args.config)
    elif args.mode == "TEST":
        print("- TESTING -")
        test(args.config)
