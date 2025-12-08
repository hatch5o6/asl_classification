import argparse
import yaml
import os
import shutil
import csv
import json

import torch
torch.set_float32_matmul_precision("medium")
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from transformers import VideoMAEImageProcessor, VideoMAEConfig
from sklearn.metrics import classification_report

from asl_dataset import RGBDSkel_Dataset
from lightning_asl import SignClassificationLightning

video_mae_config = VideoMAEConfig()

def train(config, trial=None, limit_train_batches=1.0, additional_callbacks=[]):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("WORLD_SIZE:", world_size, "RANK:", os.environ.get("RANK"))
    print("RANK (rank_zero_only.rank): ", rank_zero_only.rank)

    rank_zero_info(f"TRAINING - CONFIG:")
    for k, v in config.items():
        rank_zero_info(f"\t-{k}=`{v}`, {type(v)}")
    rank_zero_info("\n\n")
    rank_zero_info(f"limit_train_batches={limit_train_batches}")
    rank_zero_info(f"additional_callbacks={additional_callbacks}")

    L.seed_everything(config["seed"], workers=True)

    # save dir
    save_dir = config["save"]
    if trial is not None:
        save_dir = save_dir + f"_trial_{trial.number}_{trial.study.study_name}_{trial.datetime_start}"
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    logs_dir = os.path.join(save_dir, "logs")
    tb_dir = os.path.join(save_dir, "tb")
    # if int(os.environ.get("RANK", 0)) == 0:
    if rank_zero_only.rank == 0:
        parent_dir = "/".join(save_dir.split("/")[:-1])
        os.makedirs(parent_dir, exist_ok=True)
        if os.path.exists(save_dir):
            # rank_zero_info(f"deleting {save_dir}")
            print(f"deleting {save_dir}")
            shutil.rmtree(save_dir)
        # rank_zero_info(f"creating {save_dir}")
        print(f"creating {save_dir}")
        os.mkdir(save_dir)
        os.mkdir(checkpoints_dir)
        os.mkdir(logs_dir)
        os.mkdir(tb_dir)

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    modalities = tuple(config["modalities"])

    # load train dataloader
    train_dataset = RGBDSkel_Dataset(
        annotations=config["train_csv"],
        processor=processor,
        num_frames=video_mae_config.num_frames,
        modalities=modalities
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # load val dataloader
    val_dataset = RGBDSkel_Dataset(
        annotations=config["val_csv"],
        processor=processor,
        num_frames=video_mae_config.num_frames,
        modalities=modalities
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # load model
    lightning_model = SignClassificationLightning(config=config)

    # callbacks
    early_stopping = EarlyStopping(monitor="val_acc", mode="max", patience=config["early_stop"], verbose=True)
    top_k_model_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="epoch={epoch}-step={step}-val_loss={val_loss:.6f}-val_acc={val_acc:.6f}",
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
    if trial is not None:
        pruning_callback = PyTorchLightningPruningCallback(
            trial,
            monitor="val_acc"
        )
        train_callbacks.append(pruning_callback)
    train_callbacks += additional_callbacks
    logger = CSVLogger(save_dir=logs_dir)
    tb_logger = TensorBoardLogger(save_dir=tb_dir)
    

    # set parallelization
    if config["n_gpus"] > 1:
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"

    # get trainer
    trainer = L.Trainer(
        max_steps=config["max_steps"],
        val_check_interval=config["val_interval"],
        accelerator=config["device"],
        devices="auto",
        default_root_dir=save_dir,
        callbacks=train_callbacks,
        logger=[logger, tb_logger],
        deterministic=True,
        strategy=strategy,
        precision="16-mixed",
        gradient_clip_val=config["gradient_clip_val"],
        limit_train_batches=limit_train_batches
    )

    # train :)
    try:
        trainer.fit(
            lightning_model,
            train_dataloader,
            val_dataloader
        )
    except optuna.exceptions.TrialPruned:
        raise

    val_acc = trainer.callback_metrics["val_acc"]
    assert isinstance(val_acc, torch.Tensor)
    val_acc = val_acc.item()
    assert isinstance(val_acc, float)

    print(f"VAL ACC AT THE END OF TRAINING: {val_acc}")

    return val_acc


def test(config):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("WORLD_SIZE:", world_size)

    L.seed_everything(config["seed"], workers=True)

    rank_zero_info(f"TESTING - CONFIG:")
    for k, v in config.items():
        rank_zero_info(f"\t-{k}=`{v}`, {type(v)}")
    rank_zero_info("\n\n")

    # save_dir
    save_dir = config["save"]
    predictions_dir = os.path.join(save_dir, "predictions")
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    # if int(os.environ.get("RANK", 0)) == 0:
    if rank_zero_only.rank == 0:
        assert os.path.exists(save_dir)
        assert os.path.exists(checkpoints_dir)
        if os.path.exists(predictions_dir):
            rank_zero_info(f"deleting {predictions_dir}")
            shutil.rmtree(predictions_dir)
        rank_zero_info(f"creating {predictions_dir}")
        os.mkdir(predictions_dir)

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    modalities = tuple(config["modalities"])

    # load test dataloader
    test_dataset = RGBDSkel_Dataset(
        annotations=config["test_csv"],
        processor=processor,
        num_frames=video_mae_config.num_frames,
        modalities=modalities
    )
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    if config["test_checkpoint"] in [None, "None", "null"]:
        print("No checkpoint provided for testing. Will select based on val acc.")
        checkpoint_to_test = None
        best_val_acc = None
        for f in os.listdir(checkpoints_dir):
            val_acc = float(f.split(".ckpt")[0].split("-val_acc=")[1])
            if best_val_acc is None:
                assert checkpoint_to_test is None
                best_val_acc = val_acc
                checkpoint_to_test = os.path.join(checkpoints_dir, f)
            elif val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_to_test = os.path.join(checkpoints_dir, f)
        assert checkpoint_to_test is not None
        assert best_val_acc is not None
    else:
        checkpoint_to_test = config["test_checkpoint"]
        assert checkpoint_to_test.endswith(".ckpt")

    lightning_model = SignClassificationLightning.load_from_checkpoint(
        checkpoint_path=checkpoint_to_test,
        config=config
    )
    lightning_model.eval()

    trainer = L.Trainer(
        accelerator=config["device"], 
        precision="16-mixed"
    )

    batch_predictions = trainer.predict(
        lightning_model,
        dataloaders=test_dataloader
    )

    predictions = []
    for b, batch in enumerate(batch_predictions):
        predictions += batch
    labels, videos = get_labels(test_csv=config["test_csv"])

    assert len(predictions) == len(labels) == len(videos)
    pred_data = list(zip(videos, labels, predictions))

    metrics = calc_metrics(predictions, labels)

    preds_out = os.path.join(predictions_dir, checkpoint_to_test.split("/")[-1] + ".predictions.csv")
    label_dict = get_label_dict(config["class_id_csv"], lang="EN")
    with open(preds_out, "w", newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(["test_video", "label", "label_word", "prediction", "prediction_word"])
        for vid, lab, pred in pred_data:
            writer.writerow([vid, lab, label_dict[lab], pred, label_dict[pred]])

    metrics_out = os.path.join(predictions_dir, checkpoint_to_test.split("/")[-1] + ".metrics.json")
    with open(metrics_out, "w") as outf:
        outf.write(json.dumps(metrics_out, ensure_ascii=False, indent=2))


def calc_metrics(predictions, labels):
    predictions_t = torch.tensor(predictions)
    labels_t = torch.tensor(labels)
    acc = (predictions_t == labels_t).float().mean()

    report_dict = classification_report(labels, predictions, output_dict=True)
    report_dict["my_accuracy"] = acc
    return report_dict

    
def get_labels(test_csv):
    videos = []
    labels = []
    with open(test_csv, newline='') as inf:
        for row in csv.reader(inf):
            video, label = tuple(row)
            label = int(label)
            videos.append(video)
            labels.append(label)
    return labels, videos

def get_label_dict(class_csv, lang="EN"):
    assert lang in ["TR", "EN"]
    label_dict = {}
    with open(class_csv, newline='') as inf:
        rows = [tuple(r) for r in csv.reader(inf)]
        header = rows[0]
        assert header == ("ClassId", "TR", "EN")
        data = rows[1:]
        for class_id, tr_word, en_word in data:
            class_id = int(class_id)

            chosen_word = None
            if lang == "EN":
                chosen_word = en_word
            elif lang == "TR":
                chosen_word = tr_word
            assert chosen_word is not None

            assert class_id not in label_dict
            label_dict[class_id] = chosen_word
    return label_dict


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
    print("READING CONFIG:", f)
    with open(f) as inf:
        config = yaml.safe_load(inf)
    config["batch_size"] = round(config["effective_batch_size"] / config["n_gpus"])
    config["pretrained_learning_rate"] = float(config["pretrained_learning_rate"])
    config["new_learning_rate"] = float(config["new_learning_rate"])
    config["skel_learning_rate"] = float(config["skel_learning_rate"])
    config["class_learning_rate"] = float(config["class_learning_rate"])
    config["warmup_steps"] = round(0.05 * config["max_steps"])
    # rank_zero_info(f"CONFIG: {f}")
    # for k, v in config.items():
    #     rank_zero_info(f"\t-{k}=`{v}`, {type(v)}")
    # rank_zero_info("\n\n")
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
    config = read_config(args.config)
    if args.mode == "TRAIN":
        print("- TRAINING -")
        train(config)
    elif args.mode == "TEST":
        print("- TESTING -")
        test(config)
