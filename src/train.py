import argparse
import yaml
import os
import shutil
import csv
import json

import torch
torch.set_float32_matmul_precision("medium")
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from datetime import timedelta
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from transformers import VideoMAEImageProcessor, VideoMAEConfig
from sklearn.metrics import classification_report

from data.asl_dataset import RGBDSkel_Dataset
from models.lightning_asl import SignClassificationLightning

video_mae_config = VideoMAEConfig()

def train(config, trial=None, limit_train_batches=1.0, additional_callbacks=[], resume=False):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("WORLD_SIZE:", world_size, "RANK:", os.environ.get("RANK"))
    print("RANK (rank_zero_only.rank): ", rank_zero_only.rank)

    rank_zero_info(f"TRAINING - CONFIG:")
    for k, v in config.items():
        rank_zero_info(f"\t-{k}=`{v}`, {type(v)}")
    rank_zero_info("\n\n")
    rank_zero_info(f"limit_train_batches={limit_train_batches}")
    rank_zero_info(f"additional_callbacks={additional_callbacks}")
    rank_zero_info(f"resume={resume}")

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
        if resume:
            assert os.path.exists(checkpoints_dir), \
                f"Cannot resume: {checkpoints_dir} does not exist"
            print(f"RESUMING from {checkpoints_dir}")
        else:
            if os.path.exists(save_dir):
                print(f"deleting {save_dir}")
                shutil.rmtree(save_dir)
            print(f"creating {save_dir}")
            os.mkdir(save_dir)
            os.mkdir(checkpoints_dir)
            os.mkdir(logs_dir)
            os.mkdir(tb_dir)

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    modalities = tuple(config["modalities"])

    num_frames = config["num_frames"]
    if num_frames == "video_mae":
        num_frames = video_mae_config.num_frames
    else:
        assert isinstance(num_frames, int)

    # Build augmentation config for training dataset (empty dict = no augmentation)
    augment_config = {}
    if config.get("augment_skeleton", False):
        augment_config["spatial"] = True
        augment_config["scale_range"] = tuple(config.get("augment_scale_range", [0.9, 1.1]))
        augment_config["rotation_deg"] = config.get("augment_rotation_deg", 15.0)
        augment_config["translate_range"] = config.get("augment_translate_range", 0.1)
        augment_config["temporal"] = config.get("augment_temporal", True)
        augment_config["speed_range"] = tuple(config.get("augment_speed_range", [0.8, 1.2]))
        augment_config["joint_noise"] = config.get("augment_joint_noise", True)
        augment_config["noise_std"] = config.get("augment_noise_std", 0.02)
        rank_zero_info(f"Skeleton augmentation enabled: {augment_config}")

    # load train dataloader
    train_dataset = RGBDSkel_Dataset(
        annotations=config["train_csv"],
        processor=processor,
        num_frames=video_mae_config.num_frames,
        modalities=modalities,
        use_tslformer_joints=config.get("use_tslformer_joints", False),
        use_z_coord=config.get("use_z_coord", False),
        selected_joint_indices=config.get("selected_joint_indices", None),
        augment_config=augment_config,
    )

    # Signer-balanced sampling: weight each sample inversely by its signer's count
    train_shuffle = True
    train_sampler = None
    if config.get("signer_balanced_sampling", False):
        signer_map_file = config["signer_map_file"]
        rank_zero_info(f"Loading signer map from {signer_map_file}")
        with open(signer_map_file) as f:
            signer_map = json.load(f)  # {video_filename: signer_id}
        # Map each train sample to a signer via its video/skel path
        from collections import Counter
        sample_signers = []
        for rgb_path, depth_path, skel_path, label in train_dataset.annotations:
            # Extract video stem from skel_path or rgb_path
            path = skel_path if skel_path.strip() else rgb_path
            basename = os.path.basename(path)
            # Strip _landmarks.npy for skeleton files
            stem = basename.replace("_landmarks.npy", "").replace(".mp4", "")
            video_key = stem + ".mp4"
            signer = signer_map.get(video_key, "unknown")
            sample_signers.append(signer)
        signer_counts = Counter(sample_signers)
        weights = [1.0 / signer_counts[s] for s in sample_signers]
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_shuffle = False  # sampler and shuffle are mutually exclusive
        rank_zero_info(f"Signer-balanced sampling: {len(signer_counts)} signers, "
                       f"min={min(signer_counts.values())} max={max(signer_counts.values())}")

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"],
                                  shuffle=train_shuffle, sampler=train_sampler)

    # load val dataloader
    val_dataset = RGBDSkel_Dataset(
        annotations=config["val_csv"],
        processor=processor,
        num_frames=video_mae_config.num_frames,
        modalities=modalities,
        use_tslformer_joints=config.get("use_tslformer_joints", False),
        use_z_coord=config.get("use_z_coord", False),
        selected_joint_indices=config.get("selected_joint_indices", None)
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # load model
    lightning_model = SignClassificationLightning(config=config)

    # callbacks
    early_stopping = EarlyStopping(monitor="val_acc", mode="max", patience=config["early_stop"], verbose=True)
    top_k_model_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="{epoch}-{step}-{val_loss:.6f}-{val_acc:.6f}",
        save_top_k=config["save_top_k"],
        monitor="val_acc",
        mode="max"
    )
    last_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="last",
        save_last=False,
        every_n_train_steps=500,
        save_top_k=1,
        monitor="step",
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    print_callback = PrintCallback()
    train_callbacks = [
        early_stopping,
        top_k_model_checkpoint,
        last_checkpoint,
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
        strategy = DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(minutes=60)
        )
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
    ckpt_path = None
    if resume:
        last_ckpt = os.path.join(checkpoints_dir, "last.ckpt")
        assert os.path.exists(last_ckpt), f"Cannot resume: {last_ckpt} not found"
        ckpt_path = last_ckpt
        rank_zero_info(f"Resuming from checkpoint: {ckpt_path}")

    try:
        trainer.fit(
            lightning_model,
            train_dataloader,
            val_dataloader,
            ckpt_path=ckpt_path
        )
    except optuna.exceptions.TrialPruned:
        raise

    val_acc = trainer.callback_metrics["val_acc"]
    assert isinstance(val_acc, torch.Tensor)
    val_acc = val_acc.item()
    assert isinstance(val_acc, float)

    print(f"VAL ACC AT THE END OF TRAINING: {val_acc}")

    return val_acc


def _collect_logits(trainer, model, dataloader):
    """Run predict and return stacked logits (N, C) instead of argmax predictions."""
    batch_results = trainer.predict(model, dataloaders=dataloader)
    all_logits = []
    for logits, _labels in batch_results:
        all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


def _collect_labels(trainer, model, dataloader):
    """Run predict and return labels as a flat list."""
    batch_results = trainer.predict(model, dataloaders=dataloader)
    labels = []
    for _logits, batch_labels in batch_results:
        labels += batch_labels.cpu().tolist()
    return labels


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

    num_frames = config["num_frames"]
    if num_frames == "video_mae":
        num_frames = video_mae_config.num_frames
    else:
        assert isinstance(num_frames, int)

    # load test dataloader (no augmentation for standard test)
    test_dataset = RGBDSkel_Dataset(
        annotations=config["test_csv"],
        processor=processor,
        num_frames=video_mae_config.num_frames,
        modalities=modalities,
        use_tslformer_joints=config.get("use_tslformer_joints", False),
        use_z_coord=config.get("use_z_coord", False),
        selected_joint_indices=config.get("selected_joint_indices", None)
    )
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    if config["test_checkpoint"] in [None, "None", "null"]:
        print("No checkpoint provided for testing. Will select based on val acc.")
        checkpoint_to_test = None
        best_val_acc = None
        for f in os.listdir(checkpoints_dir):
            # val_acc = float(f.split(".ckpt")[0].split("-val_acc=val_acc=")[1])
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

    # ── Test-time augmentation (TTA) ──────────────────────────────────────
    tta_runs = config.get("tta_augments", 0)
    if tta_runs > 0:
        print(f"TTA enabled: {tta_runs} augmented passes + 1 clean pass")
        from data.asl_dataset import augment_skeleton_spatial, augment_skeleton_temporal, augment_skeleton_joint_noise

        # 1) Clean pass — collect logits
        all_logits = _collect_logits(trainer, lightning_model, test_dataloader)

        # 2) Augmented passes
        for tta_i in range(tta_runs):
            print(f"  TTA pass {tta_i + 1}/{tta_runs}")
            tta_aug_config = {
                "spatial": True,
                "scale_range": tuple(config.get("tta_scale_range", [0.95, 1.05])),
                "rotation_deg": config.get("tta_rotation_deg", 10.0),
                "translate_range": config.get("tta_translate_range", 0.05),
                "temporal": True,
                "speed_range": tuple(config.get("tta_speed_range", [0.9, 1.1])),
                "joint_noise": True,
                "noise_std": config.get("tta_noise_std", 0.01),
            }
            tta_dataset = RGBDSkel_Dataset(
                annotations=config["test_csv"],
                processor=processor,
                num_frames=video_mae_config.num_frames,
                modalities=modalities,
                use_tslformer_joints=config.get("use_tslformer_joints", False),
                use_z_coord=config.get("use_z_coord", False),
                selected_joint_indices=config.get("selected_joint_indices", None),
                augment_config=tta_aug_config,
            )
            tta_loader = DataLoader(tta_dataset, batch_size=config["batch_size"], shuffle=False)
            aug_logits = _collect_logits(trainer, lightning_model, tta_loader)
            all_logits = all_logits + aug_logits

        # Average logits across all passes
        avg_logits = all_logits / (1 + tta_runs)
        predictions = avg_logits.argmax(dim=1).tolist()
        pred_labels = _collect_labels(trainer, lightning_model, test_dataloader)
    else:
        # Standard single-pass prediction
        batch_predictions = trainer.predict(
            lightning_model,
            dataloaders=test_dataloader
        )
        print("BATCH PREDICTIONS")
        print(batch_predictions)

        predictions = []
        pred_labels = []
        for b, batch in enumerate(batch_predictions):
            b_logits, b_labels = batch
            b_preds = torch.argmax(b_logits, dim=1)
            predictions += b_preds.cpu().tolist()
            pred_labels += b_labels.cpu().tolist()

    print("predictions")
    print(predictions)
    print("pred labels")
    print(pred_labels)
    labels, videos = get_labels(test_csv=config["test_csv"])
    assert pred_labels == labels, f"pred_labels: {pred_labels}::::labels: {labels}"
    print("pred_labels == labels, as it should :)")

    assert len(predictions) == len(labels) == len(videos), f"{len(predictions)}, {len(labels)}, {len(videos)}"
    pred_data = list(zip(videos, labels, predictions))

    metrics = calc_metrics(predictions, labels)

    preds_out = os.path.join(predictions_dir, checkpoint_to_test.split("/")[-1] + ".predictions.csv")
    label_dict = get_label_dict(config["class_id_csv"], lang="EN")
    with open(preds_out, "w", newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(["test_video", "label", "label_word", "prediction", "prediction_word"])
        for vid, lab, pred in pred_data:
            writer.writerow([vid, lab, label_dict[lab], pred, label_dict[pred]])

    print("metrics")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    metrics_out = os.path.join(predictions_dir, checkpoint_to_test.split("/")[-1] + ".metrics.json")
    with open(metrics_out, "w") as outf:
        outf.write(json.dumps(metrics, ensure_ascii=False, indent=2))


def calc_metrics(predictions, labels):
    predictions_t = torch.tensor(predictions)
    labels_t = torch.tensor(labels)
    acc = (predictions_t == labels_t).float().mean()

    report_dict = classification_report(labels, predictions, output_dict=True)
    report_dict["my_accuracy"] = acc.item()
    return report_dict

    
def get_labels(test_csv):
    videos = []
    labels = []
    with open(test_csv, newline='') as inf:
        r = 0
        for row in csv.reader(inf):
            if r == 0:
                assert tuple(row) == ("rgb_path","depth_path","skel_path","label"), f"ROW={tuple(row)}"
                r += 1
                continue
            rgb, depth, skel, label = tuple(row)
            video = "::::".join((rgb, depth, skel))
            label = int(label)
            videos.append(video)
            labels.append(label)
            r += 1
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

    #adjust for user
    user = os.getenv("USER")
    if user:
        config_str = json.dumps(config)
        config_str = config_str.replace("hatch5o6", user)
        config = json.loads(config_str)

    config["batch_size"] = round(config["effective_batch_size"] / config["n_gpus"])
    config["pretrained_learning_rate"] = float(config["pretrained_learning_rate"])
    config["new_learning_rate"] = float(config["new_learning_rate"])
    config["skel_learning_rate"] = float(config["skel_learning_rate"])
    config["class_learning_rate"] = float(config["class_learning_rate"])
    if "gate_learning_rate" in config:
        config["gate_learning_rate"] = float(config["gate_learning_rate"])
    config["warmup_steps"] = round(0.05 * config["max_steps"])

    # Load custom joint indices if specified
    if "joint_indices_file" in config and config["joint_indices_file"] is not None:
        joint_indices_path = config["joint_indices_file"]
        print(f"Loading joint indices from: {joint_indices_path}")
        with open(joint_indices_path) as jf:
            config["selected_joint_indices"] = json.load(jf)
        print(f"  Loaded {len(config['selected_joint_indices'])} joint indices")
        assert len(config["selected_joint_indices"]) == config["num_pose_points"], \
            (f"joint_indices_file has {len(config['selected_joint_indices'])} indices "
             f"but num_pose_points is {config['num_pose_points']}")

    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-m", "--mode", choices = ["TRAIN", "TEST", "RESUME"])
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
    elif args.mode == "RESUME":
        print("- RESUMING TRAINING -")
        train(config, resume=True)
    elif args.mode == "TEST":
        print("- TESTING -")
        test(config)
